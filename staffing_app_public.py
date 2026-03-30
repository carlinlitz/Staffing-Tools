import streamlit as st
import random
from ortools.sat.python import cp_model
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Staffing Scenario Planner", layout="wide")
st.title("Staffing Scenario Planner")
st.markdown("---")

# ── Helper functions ──────────────────────────────────────────────────────────

def create_staff(n_senior, n_junior, unavailable_weeks=3, seed=None):
    if seed is not None:
        random.seed(seed)
    staff = []
    for i in range(n_senior):
        staff.append({
            "name": f"Senior_{i+1}",
            "available_from": 1,
            "type": "existing",
            "seniority": "senior",
            "unavailable": random.sample(range(1, 53), unavailable_weeks) if unavailable_weeks > 0 else []
        })
    for i in range(n_junior):
        staff.append({
            "name": f"Junior_{i+1}",
            "available_from": 1,
            "type": "existing",
            "seniority": "junior",
            "unavailable": random.sample(range(1, 53), unavailable_weeks) if unavailable_weeks > 0 else []
        })
    return staff


def build_assignments(rotation_tracks, sudden_onset_count=0, seed=1):
    """
    Regular rotation tracks: back-to-back with 1-week handover overlap.
    Sudden-onset assignments: randomized timing and duration (2-6 weeks).
    """
    rng = random.Random(seed)
    all_assignments = []

    for track in rotation_tracks:
        name     = track["name"]
        duration = track["duration"]
        diff     = track["difficulty"]
        step     = max(1, duration - 1)  # 1-week overlap
        start    = 1
        slot     = 1
        while start + duration - 1 <= 52:
            all_assignments.append({
                "name":       f"{name} {slot}",
                "start":      start,
                "duration":   duration,
                "difficulty": diff,
                "track":      name,
            })
            start += step
            slot  += 1

    for i in range(sudden_onset_count):
        duration = rng.randint(2, 6)
        start    = rng.randint(1, 52 - duration + 1)
        all_assignments.append({
            "name":       f"Sudden-Onset {i+1}",
            "start":      start,
            "duration":   duration,
            "difficulty": "high",
            "track":      "Sudden-Onset",
        })

    return all_assignments


def solve_staffing(all_assignments, staff, max_assignments_per_person=2,
                   min_gap=12, year_weeks=52, dc_floor=0):
    model = cp_model.CpModel()
    num_people      = len(staff)
    num_assignments = len(all_assignments)

    x = {}
    for p in range(num_people):
        for a in range(num_assignments):
            x[(p, a)] = model.NewBoolVar(f"x_p{p}_a{a}")

    # Each assignment must be covered by exactly one person
    for a in range(num_assignments):
        model.Add(sum(x[(p, a)] for p in range(num_people)) == 1)

    for p in range(num_people):
        person = staff[p]

        # Availability from start week
        for a in range(num_assignments):
            if all_assignments[a]["start"] < person["available_from"]:
                model.Add(x[(p, a)] == 0)

        # Block juniors from high difficulty
        if person["seniority"] == "junior":
            for a in range(num_assignments):
                if all_assignments[a]["difficulty"] == "high":
                    model.Add(x[(p, a)] == 0)

        # Unavailable weeks
        for a in range(num_assignments):
            assign_weeks = range(all_assignments[a]["start"],
                                 all_assignments[a]["start"] + all_assignments[a]["duration"])
            if any(week in person["unavailable"] for week in assign_weeks):
                model.Add(x[(p, a)] == 0)

        # Max assignments per person
        model.Add(sum(x[(p, a)] for a in range(num_assignments)) <= max_assignments_per_person)

        # No overlap + minimum gap between assignments
        for a1 in range(num_assignments):
            for a2 in range(a1 + 1, num_assignments):
                start1 = all_assignments[a1]["start"]
                end1   = start1 + all_assignments[a1]["duration"]
                start2 = all_assignments[a2]["start"]
                end2   = start2 + all_assignments[a2]["duration"]
                if not (end1 + min_gap <= start2 or end2 + min_gap <= start1):
                    model.Add(x[(p, a1)] + x[(p, a2)] <= 1)

        # Annual capacity
        total_weeks = sum(x[(p, a)] * all_assignments[a]["duration"] for a in range(num_assignments))
        model.Add(total_weeks <= year_weeks)

    # DC floor constraint
    if dc_floor > 0:
        senior_indices = [p for p in range(num_people) if staff[p]["seniority"] == "senior"]
        for week in range(1, year_weeks + 1):
            active_senior_indices = senior_indices  # no resignations in public version
            seniors_available = len(active_senior_indices)
            max_away = max(0, seniors_available - max(0, dc_floor - 2))
            active_assignments = [
                a for a in range(num_assignments)
                if all_assignments[a]["start"] <= week < all_assignments[a]["start"] + all_assignments[a]["duration"]
            ]
            if active_assignments:
                seniors_away = sum(x[(p, a)] for p in active_senior_indices for a in active_assignments)
                model.Add(seniors_away <= max_away)

    used = []
    for p in range(num_people):
        u = model.NewBoolVar(f"used_{p}")
        model.AddMaxEquality(u, [x[(p, a)] for a in range(num_assignments)])
        used.append(u)
    model.Minimize(sum(used))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    result = solver.Solve(model)

    if result not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, solver.StatusName(result)

    results = []
    for p in range(num_people):
        if solver.Value(used[p]):
            person = staff[p].copy()
            person["assignments"] = [
                all_assignments[a]["name"]
                for a in range(num_assignments)
                if solver.Value(x[(p, a)])
            ]
            results.append(person)

    return results, solver.StatusName(result)


# ── Gantt chart ───────────────────────────────────────────────────────────────

colors = {
    "high":          "#c9a84c",
    "moderate":      "#7eb8c9",
    "unavailable":   "#5a2d2d",
    "not_available": "#1e1e2e",
    "free":          "#2a2a3a",
}

def build_gantt(results, all_assignments, people, staff, dc_floor=0, show_dc_band=False):
    assignment_map = {a["name"]: a for a in all_assignments}

    fig = go.Figure()

    for person in people:
        label = f"{person['name']} ({person['seniority']})"
        person_result = next((r for r in results if r["name"] == person["name"]), None)
        assigned_names = person_result["assignments"] if person_result else []

        for week in range(1, 53):
            if week < person["available_from"]:
                color   = colors["not_available"]
                tooltip = f"Week {week}: Not yet available"
            elif week in person.get("unavailable", []):
                color   = colors["unavailable"]
                tooltip = f"Week {week}: Unavailable"
            else:
                assigned = None
                for aname in assigned_names:
                    a = assignment_map.get(aname)
                    if a and a["start"] <= week < a["start"] + a["duration"]:
                        assigned = a
                        break
                if assigned:
                    color   = colors[assigned["difficulty"]]
                    tooltip = f"Week {week}: {assigned['name']} ({assigned['difficulty']})"
                else:
                    color   = colors["free"]
                    tooltip = f"Week {week}: Free"

            fig.add_trace(go.Bar(
                x=[1], y=[label], base=[week - 1],
                orientation="h",
                marker_color=color,
                hovertemplate=f"<b>{label}</b><br>{tooltip}<extra></extra>",
                showlegend=False,
            ))

    month_starts = [1, 5, 9, 14, 18, 22, 27, 31, 35, 40, 44, 48]
    months       = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ms, m in zip(month_starts, months):
        fig.add_vline(x=ms - 1, line_width=1, line_color="#3a3a4a")
        fig.add_annotation(x=ms - 0.5, y=1.02, text=m, showarrow=False,
                           font=dict(size=9, color="#888"), yref="paper", xref="x")

    # DC floor band
    if dc_floor > 0 and show_dc_band:
        all_seniors     = [p for p in staff if p["seniority"] == "senior"]
        senior_results  = [p for p in results if p["seniority"] == "senior"]
        at_floor_weeks  = []
        for week in range(1, 53):
            active_seniors = len(all_seniors)
            on_assignment  = sum(
                1 for p in senior_results
                if any(
                    assignment_map[aname]["start"] <= week < assignment_map[aname]["start"] + assignment_map[aname]["duration"]
                    for aname in p["assignments"] if aname in assignment_map
                )
            )
            dc_headcount = active_seniors - on_assignment
            if dc_headcount <= dc_floor:
                at_floor_weeks.append(week)

        for week in at_floor_weeks:
            fig.add_shape(
                type="rect",
                x0=week - 1, x1=week,
                y0=-0.5, y1=-0.05,
                yref="paper",
                fillcolor="rgba(232, 115, 74, 0.4)",
                line_width=0,
            )
        if at_floor_weeks:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(color="rgba(232,115,74,0.6)", symbol="square", size=10),
                name=f"DC at/below floor ({int(dc_floor)})",
                showlegend=True,
            ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0f0f14", plot_bgcolor="#0f0f14",
        font=dict(color="#e8e4d9", family="Georgia"),
        xaxis=dict(range=[0, 52], title="Week", tickfont=dict(size=10), gridcolor="#2a2a3a"),
        yaxis=dict(title="", tickfont=dict(size=11)),
        height=max(400, len(people) * 40 + 100),
        margin=dict(l=160, r=20, t=40, b=40),
        hoverlabel=dict(bgcolor="#1a1a24", font_size=12),
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Controls")

st.sidebar.subheader("👥 Staff Composition")
n_senior          = st.sidebar.number_input("Number of Seniors",  0, 30, 14)
n_junior          = st.sidebar.number_input("Number of Juniors",  0, 30, 0)
unavailable_weeks = st.sidebar.slider("Unavailable Weeks per Person", 0, 8, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Rotation Tracks")
st.sidebar.caption("Define each rotation type. Slots are spread evenly across the year.")

num_tracks = st.sidebar.number_input("Number of rotation track types", 1, 8, 2)
st.sidebar.caption("Each track runs back-to-back all year with a 1-week handover overlap.")

rotation_tracks = []
for i in range(int(num_tracks)):
    with st.sidebar.expander(f"Track {i+1}", expanded=(i < 2)):
        t_name     = st.text_input(f"Track name",         value=f"Rotation {i+1}",  key=f"tname_{i}")
        t_duration = st.number_input(f"Duration (weeks)", min_value=2, max_value=26, value=6 if i == 0 else 8, key=f"tdur_{i}")
        t_diff     = st.selectbox(f"Difficulty",          ["high", "moderate"],      index=0, key=f"tdiff_{i}")
        rotation_tracks.append({
            "name":       t_name,
            "duration":   int(t_duration),
            "difficulty": t_diff,
        })

st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Sudden-Onset Assignments")
st.sidebar.caption("Randomized timing and duration (2–6 weeks). Always high difficulty.")
sudden_onset_count = st.sidebar.number_input("Number of sudden-onset assignments this year", 0, 20, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Settings")
max_assignments = st.sidebar.slider("Max Assignments per Person", 1, 4, 2)
min_gap         = st.sidebar.slider("Minimum Gap Between Assignments (weeks)", 4, 20, 12)
seed            = st.sidebar.slider("Randomization Seed", 1, 100, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("🏛️ DC Operations")
dc_floor = st.sidebar.number_input("Minimum Seniors in DC at All Times", 0, 20, 0,
               help="Seniors on assignment cannot drop DC headcount below this number")

# ── Build and solve ───────────────────────────────────────────────────────────

all_assignments = build_assignments(rotation_tracks, int(sudden_onset_count), seed=seed)
staff           = create_staff(int(n_senior), int(n_junior),
                               unavailable_weeks=unavailable_weeks, seed=seed)

total_slots = len(all_assignments)
rotation_desc = "  ·  ".join(
    f"{t['duration']}-wk {t['name']}"
    for t in rotation_tracks
)
sudden_desc = f"{int(sudden_onset_count)} sudden-onset" if sudden_onset_count > 0 else ""
scenario_desc = "  ·  ".join(filter(None, [rotation_desc, sudden_desc]))

if dc_floor > 0 and dc_floor >= int(n_senior):
    st.warning(f"⚠️ DC floor ({int(dc_floor)}) is equal to or greater than total seniors ({int(n_senior)}) — no one could go on assignment. Reduce the DC floor.")

if total_slots == 0:
    st.warning("No assignments defined — add at least one rotation track or sudden-onset assignment to run the solver.")
    st.stop()

with st.spinner("Solving..."):
    results, status = solve_staffing(
        all_assignments, staff,
        max_assignments_per_person=max_assignments,
        min_gap=min_gap,
        dc_floor=int(dc_floor)
    )

st.subheader(f"Scenario: {scenario_desc}")
st.caption(f"Solver status: {status}  ·  {total_slots} total assignment slots")

if results is None:
    st.error("No feasible solution found. Try adjusting the controls — reduce DC floor, add more staff, or reduce assignment slots.")
else:
    # Metrics
    assignment_map = {a["name"]: a for a in all_assignments}
    all_seniors    = [p for p in staff if p["seniority"] == "senior"]
    senior_results = [p for p in results if p["seniority"] == "senior"]
    dc_by_week     = []
    for week in range(1, 53):
        on_assignment = sum(
            1 for p in senior_results
            if any(
                assignment_map[aname]["start"] <= week < assignment_map[aname]["start"] + assignment_map[aname]["duration"]
                for aname in p["assignments"] if aname in assignment_map
            )
        )
        dc_by_week.append(len(all_seniors) - on_assignment)
    min_dc = min(dc_by_week) if dc_by_week else int(n_senior)

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Staff Required",    len(results))
    col2.metric("Total Assignments", len(all_assignments))
    col3.metric("Seniors Used",      sum(1 for p in results if p["seniority"] == "senior"))
    col4.metric("Juniors Used",      sum(1 for p in results if p["seniority"] == "junior"))
    col5.metric("New Hires Used",    sum(1 for p in results if p["type"] == "new_hire"))
    col6.metric("Min DC Headcount",  min_dc,
                delta=f"{min_dc - int(dc_floor)} above floor" if dc_floor > 0 else None,
                delta_color="normal")

    if dc_floor > 0 and min_dc < dc_floor - 2:
        st.error(f"🚨 Critical DC breach — minimum DC headcount ({min_dc}) fell more than 2 below the floor ({int(dc_floor)}).")
    elif dc_floor > 0 and min_dc < dc_floor:
        st.warning(f"⚠️ DC floor breached — minimum DC headcount was {min_dc} (floor: {int(dc_floor)}).")
    elif dc_floor > 0:
        st.success(f"✅ DC floor maintained — minimum DC headcount was {min_dc} (floor: {int(dc_floor)})")

    st.markdown("---")
    st.subheader("Schedule")

    lcol1, lcol2, lcol3, lcol4, lcol5 = st.columns(5)
    lcol1.markdown(f'<span style="color:{colors["high"]}">■</span> High difficulty',          unsafe_allow_html=True)
    lcol2.markdown(f'<span style="color:{colors["moderate"]}">■</span> Moderate difficulty',  unsafe_allow_html=True)
    lcol3.markdown(f'<span style="color:{colors["unavailable"]}">■</span> Unavailable / Leave', unsafe_allow_html=True)
    lcol4.markdown(f'<span style="color:{colors["not_available"]}">■</span> Not yet hired',   unsafe_allow_html=True)
    lcol5.markdown(f'<span style="color:{colors["free"]}">■</span> Free / Available',         unsafe_allow_html=True)

    result_names = {p["name"] for p in results}

    def sort_by_role(p):
        return ({"senior": 0, "junior": 1}.get(p["seniority"], 2), p["name"])

    used_staff   = sorted([p for p in staff if p["name"] in result_names],  key=sort_by_role)
    unused_staff = sorted([p for p in staff if p["name"] not in result_names], key=sort_by_role)

    st.markdown("##### Deployed Staff")
    fig1 = build_gantt(results, all_assignments, used_staff, staff,
                       dc_floor=int(dc_floor), show_dc_band=True)
    st.plotly_chart(fig1, use_container_width=True)

    if unused_staff:
        st.markdown("##### Reserve / Not Needed for This Scenario")
        fig2 = build_gantt(results, all_assignments, unused_staff, staff,
                           dc_floor=0, show_dc_band=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Assignment Details")
    table_data = []
    for p in results:
        table_data.append({
            "Name":           p["name"],
            "Seniority":      p["seniority"].capitalize(),
            "Type":           p["type"].replace("_", " ").title(),
            "Available From": f"Week {p['available_from']}",
            "Assignments":    ", ".join(p["assignments"]) if p["assignments"] else "None",
        })
    st.dataframe(table_data, use_container_width=True)
