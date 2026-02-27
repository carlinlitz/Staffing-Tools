import streamlit as st
import random
from ortools.sat.python import cp_model
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Staffing Scenario Planner", layout="wide")
st.title("Staffing Scenario Planner")
st.markdown("---")

# ── Helper functions ──────────────────────────────────────────────────────────

def create_staff(n_senior, n_junior,
                 senior_resignations=0, junior_resignations=0,
                 new_hires=0, new_hire_delay=39, new_hire_seniority="junior",
                 unavailable_weeks=3, seed=None):
    if seed is not None:
        random.seed(seed)

    staff = []

    senior_resign_weeks = random.sample(range(1, 53), senior_resignations) if senior_resignations > 0 else []
    for i in range(n_senior):
        entry = {
            "name": f"Senior_{i+1}",
            "available_from": 1,
            "type": "existing",
            "seniority": "senior",
            "unavailable": random.sample(range(1, 53), unavailable_weeks) if unavailable_weeks > 0 else []
        }
        if i < senior_resignations:
            entry["resign_week"] = senior_resign_weeks[i]
        staff.append(entry)

    junior_resign_weeks = random.sample(range(1, 53), junior_resignations) if junior_resignations > 0 else []
    for i in range(n_junior):
        entry = {
            "name": f"Junior_{i+1}",
            "available_from": 1,
            "type": "existing",
            "seniority": "junior",
            "unavailable": random.sample(range(1, 53), unavailable_weeks) if unavailable_weeks > 0 else []
        }
        if i < junior_resignations:
            entry["resign_week"] = junior_resign_weeks[i]
        staff.append(entry)

    # Replacement hires — available after resign week + onboarding delay
    all_resign_weeks = senior_resign_weeks + junior_resign_weeks
    for i, week in enumerate(all_resign_weeks):
        available_from = min(week + new_hire_delay, 52)
        weeks_left = 52 - available_from
        staff.append({
            "name": f"Replacement_Hire_{i+1}",
            "available_from": available_from,
            "type": "new_hire",
            "seniority": new_hire_seniority,
            "unavailable": random.sample(range(available_from, 53), min(unavailable_weeks, max(1, weeks_left))) if unavailable_weeks > 0 and weeks_left > 0 else []
        })

    # Additional growth hires
    for i in range(new_hires):
        weeks_left = 52 - new_hire_delay
        staff.append({
            "name": f"New_Hire_{i+1}",
            "available_from": new_hire_delay,
            "type": "new_hire",
            "seniority": new_hire_seniority,
            "unavailable": random.sample(range(new_hire_delay, 53), min(unavailable_weeks, max(1, weeks_left))) if unavailable_weeks > 0 and weeks_left > 0 else []
        })

    return staff


def solve_staffing(all_assignments, staff, max_assignments_per_person=2, min_gap=12, year_weeks=52, dc_floor=0):
    model = cp_model.CpModel()
    num_people = len(staff)
    num_assignments = len(all_assignments)

    x = {}
    for p in range(num_people):
        for a in range(num_assignments):
            x[(p, a)] = model.NewBoolVar(f"x_p{p}_a{a}")

    for a in range(num_assignments):
        model.Add(sum(x[(p, a)] for p in range(num_people)) == 1)

    for p in range(num_people):
        person = staff[p]

        # Availability
        for a in range(num_assignments):
            if all_assignments[a]["start"] < person["available_from"]:
                model.Add(x[(p, a)] == 0)

        # Resignation — block assignments after OR overlapping with resign week
        if "resign_week" in person:
            for a in range(num_assignments):
                assign_end = all_assignments[a]["start"] + all_assignments[a]["duration"]
                if all_assignments[a]["start"] >= person["resign_week"]:
                    model.Add(x[(p, a)] == 0)
                elif all_assignments[a]["start"] < person["resign_week"] < assign_end:
                    model.Add(x[(p, a)] == 0)

        # Block junior/new hires from high difficulty
        if staff[p]["seniority"] == "junior":
            for a in range(num_assignments):
                if all_assignments[a]["difficulty"] == "high":
                    model.Add(x[(p, a)] == 0)

        # Unavailable weeks
        for a in range(num_assignments):
            assign_weeks = range(all_assignments[a]["start"],
                                 all_assignments[a]["start"] + all_assignments[a]["duration"])
            if any(week in person["unavailable"] for week in assign_weeks):
                model.Add(x[(p, a)] == 0)

        # Max assignments
        model.Add(sum(x[(p, a)] for a in range(num_assignments)) <= max_assignments_per_person)

        # No overlap + min gap
        for a1 in range(num_assignments):
            for a2 in range(a1 + 1, num_assignments):
                start1 = all_assignments[a1]["start"]
                end1 = start1 + all_assignments[a1]["duration"]
                start2 = all_assignments[a2]["start"]
                end2 = start2 + all_assignments[a2]["duration"]
                if not (end1 + min_gap <= start2 or end2 + min_gap <= start1):
                    model.Add(x[(p, a1)] + x[(p, a2)] <= 1)

        # Annual capacity
        total_weeks = sum(x[(p, a)] * all_assignments[a]["duration"] for a in range(num_assignments))
        model.Add(total_weeks <= year_weeks)

    # DC floor constraint — for each week, account for resignations to get actual available seniors
    if dc_floor > 0:
        senior_indices = [p for p in range(num_people) if staff[p]["seniority"] == "senior"]
        for week in range(1, year_weeks + 1):
            # Seniors still active this week (not yet resigned)
            active_senior_indices = [
                p for p in senior_indices
                if not ("resign_week" in staff[p] and staff[p]["resign_week"] <= week)
            ]
            seniors_available = len(active_senior_indices)
            max_away = max(0, seniors_available - max(0, dc_floor - 2))  # hard limit is 2 below floor
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


# ── Scenario definitions ──────────────────────────────────────────────────────

def get_scenario(scenario_name, seed, n_senior, n_junior,
                 senior_resignations, junior_resignations,
                 new_hires, new_hire_delay,
                 rotation_6wk_difficulty, concurrent_6wk_difficulty,
                 rotation_8wk_difficulty, concurrent_8wk_difficulty,
                 special_difficulty, unavailable_weeks):
    all_assignments = []

    if scenario_name == "Scenario 1":
        # 10 six-week rotations, 2 special assignments
        start = 1
        for i in range(10):
            all_assignments.append({"name": f"Rotation {i+1}", "start": start, "duration": 6, "difficulty": rotation_6wk_difficulty})
            start += 5
        all_assignments += [
            {"name": "Special A", "start": 8, "duration": 4, "difficulty": special_difficulty},
            {"name": "Special B", "start": 22, "duration": 4, "difficulty": special_difficulty},
        ]

    elif scenario_name == "Scenario 2":
        # 10 six-week rotations, 4 special assignments
        start = 1
        for i in range(10):
            all_assignments.append({"name": f"Rotation {i+1}", "start": start, "duration": 6, "difficulty": rotation_6wk_difficulty})
            start += 5
        all_assignments += [
            {"name": "Special A", "start": 8, "duration": 4, "difficulty": special_difficulty},
            {"name": "Special B", "start": 22, "duration": 4, "difficulty": special_difficulty},
            {"name": "Special C", "start": 35, "duration": 4, "difficulty": special_difficulty},
            {"name": "Special D", "start": 46, "duration": 4, "difficulty": special_difficulty},
        ]

    elif scenario_name == "Scenario 3":
        # 10 six-week + 7 eight-week rotations, 2 special assignments
        start = 1
        for i in range(10):
            all_assignments.append({"name": f"Rotation_6wk_{i+1}", "start": start, "duration": 6, "difficulty": rotation_6wk_difficulty})
            start += 5
        start = 2
        for i in range(7):
            all_assignments.append({"name": f"Rotation_8wk_{i+1}", "start": start, "duration": 8, "difficulty": rotation_8wk_difficulty})
            start += 7
        all_assignments += [
            {"name": "Special A", "start": 8, "duration": 4, "difficulty": special_difficulty},
            {"name": "Special B", "start": 22, "duration": 4, "difficulty": special_difficulty},
        ]

    elif scenario_name == "Scenario 4":
        # 14 eight-week rotations (2 concurrent), 2 special assignments
        start = 1
        for i in range(7):
            all_assignments.append({"name": f"Rotation {i+1}", "start": start, "duration": 8, "difficulty": rotation_8wk_difficulty})
            all_assignments.append({"name": f"Rotation {i+8}", "start": start + 1, "duration": 8, "difficulty": concurrent_8wk_difficulty})
            start += 7
        all_assignments += [
            {"name": "Special A", "start": 8, "duration": 4, "difficulty": special_difficulty},
            {"name": "Special B", "start": 22, "duration": 4, "difficulty": special_difficulty},
        ]

    elif scenario_name == "Scenario 5":
        # 20 six-week rotations (2 concurrent), 2 special assignments
        start = 1
        for i in range(10):
            all_assignments.append({"name": f"Rotation {i+1}", "start": start, "duration": 6, "difficulty": rotation_6wk_difficulty})
            all_assignments.append({"name": f"Rotation {i+11}", "start": start, "duration": 6, "difficulty": concurrent_6wk_difficulty})
            start += 5
        all_assignments += [
            {"name": "Special A", "start": 8, "duration": 4, "difficulty": special_difficulty},
            {"name": "Special B", "start": 22, "duration": 4, "difficulty": special_difficulty},
        ]

    staff = create_staff(
        n_senior=n_senior, n_junior=n_junior,
        senior_resignations=senior_resignations,
        junior_resignations=junior_resignations,
        new_hires=new_hires, new_hire_delay=new_hire_delay,
        unavailable_weeks=unavailable_weeks, seed=seed
    )

    return all_assignments, staff


# ── Gantt chart ───────────────────────────────────────────────────────────────

def build_gantt(results, all_assignments, people, staff, dc_floor=0, show_dc_band=False):
    """
    results: solver output (used for assignment lookup)
    people: list of staff to render in this chart
    staff: full staff list (used for DC floor calc)
    """
    assignment_map = {a["name"]: a for a in all_assignments}
    colors = {
        "high":          "#c9a84c",
        "moderate":      "#7eb8c9",
        "unavailable":   "#5a2d2d",
        "not_available": "#1e1e2e",
        "resigned":      "#2d2d5a",
        "free":          "#2a2a3a",
    }
    result_names = {p["name"] for p in results}

    fig = go.Figure()

    for person in people:
        label = f"{person['name']} ({person['seniority']})"
        person_result = next((r for r in results if r["name"] == person["name"]), None)
        assigned_names = person_result["assignments"] if person_result else []

        for week in range(1, 53):
            if week < person["available_from"]:
                color = colors["not_available"]
                tooltip = f"Week {week}: Not yet hired"
            elif "resign_week" in person and week >= person["resign_week"]:
                color = colors["resigned"]
                tooltip = f"Week {week}: Resigned (week {person['resign_week']})"
            elif week in person.get("unavailable", []):
                color = colors["unavailable"]
                tooltip = f"Week {week}: Unavailable"
            else:
                assigned = None
                for aname in assigned_names:
                    a = assignment_map.get(aname)
                    if a and a["start"] <= week < a["start"] + a["duration"]:
                        assigned = a
                        break
                if assigned:
                    color = colors[assigned["difficulty"]]
                    tooltip = f"Week {week}: {assigned['name']} ({assigned['difficulty']})"
                else:
                    color = colors["free"]
                    tooltip = f"Week {week}: Free"

            fig.add_trace(go.Bar(
                x=[1], y=[label], base=[week - 1],
                orientation="h",
                marker_color=color,
                hovertemplate=f"<b>{label}</b><br>{tooltip}<extra></extra>",
                showlegend=False,
            ))

    month_starts = [1, 5, 9, 14, 18, 22, 27, 31, 35, 40, 44, 48]
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for ms, m in zip(month_starts, months):
        fig.add_vline(x=ms - 1, line_width=1, line_color="#3a3a4a")
        fig.add_annotation(x=ms - 0.5, y=1.02, text=m, showarrow=False,
                           font=dict(size=9, color="#888"), yref="paper", xref="x")

    # DC floor shaded band — highlight weeks where seniors in DC hit the floor
    if dc_floor > 0 and show_dc_band:
        assignment_map_gantt = {a["name"]: a for a in all_assignments}
        senior_results = [p for p in results if p["seniority"] == "senior"]
        all_seniors_gantt = [p for p in staff if p["seniority"] == "senior"]
        at_floor_weeks = []
        for week in range(1, 53):
            # Account for resignations
            active_seniors_this_week = sum(
                1 for p in all_seniors_gantt
                if not ("resign_week" in p and p["resign_week"] <= week)
            )
            on_assignment = sum(
                1 for p in senior_results
                if any(
                    assignment_map_gantt[aname]["start"] <= week < assignment_map_gantt[aname]["start"] + assignment_map_gantt[aname]["duration"]
                    for aname in p["assignments"] if aname in assignment_map_gantt
                )
            )
            dc_headcount = active_seniors_this_week - on_assignment
            if dc_headcount <= dc_floor:
                at_floor_weeks.append(week)

        # Add subtle shaded rectangles for at-floor weeks
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
            # Add a single legend entry for the band
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color="rgba(232,115,74,0.6)", symbol="square", size=10),
                name=f"DC at/below floor ({int(dc_floor)})",
                showlegend=True,
            ))

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#0f0f14",
        plot_bgcolor="#0f0f14",
        font=dict(color="#e8e4d9", family="Georgia"),
        xaxis=dict(range=[0, 52], title="Week", tickfont=dict(size=10), gridcolor="#2a2a3a"),
        yaxis=dict(title="", tickfont=dict(size=11)),
        height=max(400, len(people) * 40 + 100),
        margin=dict(l=160, r=20, t=40, b=40),
        hoverlabel=dict(bgcolor="#1a1a24", font_size=12),
    )
    return fig


# ── Sidebar controls ──────────────────────────────────────────────────────────

st.sidebar.header("Controls")

descriptions = {
    "Scenario 1": "10 × 6-week rotations, 2 special assignments",
    "Scenario 2": "10 × 6-week rotations, 4 special assignments",
    "Scenario 3": "10 × 6-week + 7 × 8-week rotations, 2 special",
    "Scenario 4": "14 × 8-week rotations (2 concurrent), 2 special",
    "Scenario 5": "20 × 6-week rotations (2 concurrent), 2 special",
}

scenario_name = st.sidebar.selectbox("Select Scenario",
    ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 4", "Scenario 5"])
st.sidebar.info(descriptions[scenario_name])

st.sidebar.markdown("---")
st.sidebar.subheader("👥 Staff Composition")
n_senior            = st.sidebar.number_input("Number of Seniors", 0, 30, 14)
n_junior            = st.sidebar.number_input("Number of Juniors", 0, 30, 0)
senior_resignations = st.sidebar.number_input("Senior Resignations", 0, int(n_senior), min(6, int(n_senior)))
junior_resignations = st.sidebar.number_input("Junior Resignations", 0, max(int(n_junior), 1), 0)
new_hires           = st.sidebar.number_input("Additional New Hires (growth)", 0, 20, 6)
new_hire_delay      = st.sidebar.slider("New Hire Start Week", 1, 51, 39)
unavailable_weeks   = st.sidebar.slider("Unavailable Weeks per Person", 0, 8, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Assignment Settings")

# Show difficulty controls based on which rotation types exist in the selected scenario
has_6wk = scenario_name in ["Scenario 1", "Scenario 2", "Scenario 3", "Scenario 5"]
has_8wk = scenario_name in ["Scenario 3", "Scenario 4"]

if has_6wk:
    rotation_6wk_difficulty = st.sidebar.selectbox("6-week Rotation Difficulty", ["high", "moderate"], index=0)
else:
    rotation_6wk_difficulty = "high"

if scenario_name == "Scenario 5":
    concurrent_6wk_difficulty = st.sidebar.selectbox("Concurrent 6-week Rotation Difficulty", ["high", "moderate"], index=1)
else:
    concurrent_6wk_difficulty = rotation_6wk_difficulty

if has_8wk:
    rotation_8wk_difficulty = st.sidebar.selectbox("8-week Rotation Difficulty", ["high", "moderate"], index=0)
else:
    rotation_8wk_difficulty = "moderate"

if scenario_name == "Scenario 4":
    concurrent_8wk_difficulty = st.sidebar.selectbox("Concurrent 8-week Rotation Difficulty", ["high", "moderate"], index=1)
else:
    concurrent_8wk_difficulty = rotation_8wk_difficulty

special_difficulty = st.sidebar.selectbox("Special Assignment Difficulty", ["high", "moderate"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Settings")
max_assignments = st.sidebar.slider("Max Assignments per Person", 1, 4, 2)
min_gap         = st.sidebar.slider("Minimum Gap Between Assignments (weeks)", 4, 20, 12)
seed            = st.sidebar.slider("Randomization Seed", 1, 100, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("🏛️ DC Operations")
dc_floor        = st.sidebar.number_input("Minimum Seniors in DC at All Times", 0, 20, 0,
                    help="Seniors on assignment cannot drop DC headcount below this number")

# ── Main content ──────────────────────────────────────────────────────────────

all_assignments, staff = get_scenario(
    scenario_name=scenario_name, seed=seed,
    n_senior=int(n_senior), n_junior=int(n_junior),
    senior_resignations=int(senior_resignations),
    junior_resignations=int(junior_resignations),
    new_hires=int(new_hires), new_hire_delay=new_hire_delay,
    rotation_6wk_difficulty=rotation_6wk_difficulty,
    concurrent_6wk_difficulty=concurrent_6wk_difficulty,
    rotation_8wk_difficulty=rotation_8wk_difficulty,
    concurrent_8wk_difficulty=concurrent_8wk_difficulty,
    special_difficulty=special_difficulty,
    unavailable_weeks=unavailable_weeks,
)

# Pre-check: warn if DC floor is impossible to satisfy
total_seniors = int(n_senior)
if dc_floor > 0 and dc_floor >= total_seniors:
    st.warning(f"⚠️ DC floor ({int(dc_floor)}) is equal to or greater than total seniors ({total_seniors}) — no one could ever go on assignment. Reduce the DC floor.")

with st.spinner("Solving..."):
    results, status = solve_staffing(all_assignments, staff,
                                     max_assignments_per_person=max_assignments,
                                     min_gap=min_gap,
                                     dc_floor=int(dc_floor))

st.subheader(f"{scenario_name} — {descriptions[scenario_name]}")
st.caption(f"Solver status: {status}")

if results is None:
    st.error("No feasible solution found. Try adjusting the controls in the sidebar.")
else:
    seniors_used = sum(1 for p in results if p["seniority"] == "senior")
    total_seniors_on_team = int(n_senior)  # full team, not just solver minimum

    # Calculate minimum DC seniors across all weeks accounting for resignations
    assignment_map = {a["name"]: a for a in all_assignments}
    all_seniors = [p for p in staff if p["seniority"] == "senior"]
    senior_results = [p for p in results if p["seniority"] == "senior"]
    dc_by_week = []
    for week in range(1, 53):
        # Seniors still on team this week (not yet resigned)
        active_seniors_this_week = sum(
            1 for p in all_seniors
            if not ("resign_week" in p and p["resign_week"] <= week)
        )
        # Seniors on international assignment this week
        on_assignment = sum(
            1 for p in senior_results
            if any(
                assignment_map[aname]["start"] <= week < assignment_map[aname]["start"] + assignment_map[aname]["duration"]
                for aname in p["assignments"] if aname in assignment_map
            )
        )
        dc_by_week.append(active_seniors_this_week - on_assignment)
    min_dc_headcount = min(dc_by_week) if dc_by_week else total_seniors_on_team

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Staff Required", len(results))
    col2.metric("Total Assignments", len(all_assignments))
    col3.metric("Seniors Used", seniors_used)
    col4.metric("Juniors Used", sum(1 for p in results if p["seniority"] == "junior"))
    col5.metric("New Hires Used", sum(1 for p in results if p["type"] == "new_hire"))
    col6.metric("Min DC Headcount", min_dc_headcount,
                delta=f"{min_dc_headcount - int(dc_floor)} above floor" if dc_floor > 0 else None,
                delta_color="normal")

    if dc_floor > 0 and min_dc_headcount < dc_floor - 2:
        st.error(f"🚨 Critical DC breach — minimum DC headcount ({min_dc_headcount}) fell more than 2 below the floor ({int(dc_floor)}). This scenario is infeasible under current constraints.")
    elif dc_floor > 0 and min_dc_headcount < dc_floor:
        st.warning(f"⚠️ DC floor breached — this scenario is only feasible if we accept a minimum DC headcount of {min_dc_headcount} (floor: {int(dc_floor)}, breach of {int(dc_floor) - min_dc_headcount}). Consider increasing staff or reducing assignments.")
    elif dc_floor > 0:
        st.success(f"✅ DC floor maintained — minimum DC headcount was {min_dc_headcount} (floor: {int(dc_floor)})")

    st.markdown("---")
    st.subheader("Schedule")

    lcol1, lcol2, lcol3, lcol4, lcol5, lcol6 = st.columns(6)
    lcol1.markdown(f'<span style="color:{colors["high"]}">■</span> High difficulty', unsafe_allow_html=True)
    lcol2.markdown(f'<span style="color:{colors["moderate"]}">■</span> Moderate difficulty', unsafe_allow_html=True)
    lcol3.markdown(f'<span style="color:{colors["unavailable"]}">■</span> Unavailable / Leave', unsafe_allow_html=True)
    lcol4.markdown(f'<span style="color:{colors["not_available"]}">■</span> Not yet hired', unsafe_allow_html=True)
    lcol5.markdown(f'<span style="color:{colors["resigned"]}">■</span> Resigned', unsafe_allow_html=True)
    lcol6.markdown(f'<span style="color:{colors["free"]}">■</span> Free / Available', unsafe_allow_html=True)

    # Split staff into used and unused
    result_names = {p["name"] for p in results}

    def sort_by_role(p):
        return ({"senior": 0, "junior": 1}.get(p["seniority"], 2), p["name"])

    used_staff   = sorted([p for p in staff if p["name"] in result_names], key=sort_by_role)
    unused_staff = sorted([p for p in staff if p["name"] not in result_names], key=sort_by_role)

    st.markdown("##### Deployed Staff")
    fig1 = build_gantt(results, all_assignments, used_staff, staff, dc_floor=int(dc_floor), show_dc_band=True)
    st.plotly_chart(fig1, use_container_width=True)

    if unused_staff:
        st.markdown("##### Reserve / Not Needed for This Scenario")
        fig2 = build_gantt(results, all_assignments, unused_staff, staff, dc_floor=0, show_dc_band=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Assignment Details")
    table_data = []
    for p in results:
        resign_info = f"Week {p['resign_week']}" if "resign_week" in p else "—"
        table_data.append({
            "Name": p["name"],
            "Seniority": p["seniority"].capitalize(),
            "Type": p["type"].replace("_", " ").title(),
            "Available From": f"Week {p['available_from']}",
            "Resigns": resign_info,
            "Assignments": ", ".join(p["assignments"]) if p["assignments"] else "None",
        })
    st.dataframe(table_data, use_container_width=True)
