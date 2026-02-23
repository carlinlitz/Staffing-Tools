import streamlit as st
import random
import plotly.graph_objects as go
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Workforce Pipeline Simulator", layout="wide")
st.title("Workforce Pipeline Simulator")
st.markdown("*Modeling attrition, promotion eligibility, and team composition over time*")
st.markdown("---")

# ── Simulation engine ─────────────────────────────────────────────────────────

def run_simulation(
    juniors_already_eligible,
    juniors_eligible_in_weeks,
    total_team_size,
    initial_seniors,
    senior_cap_pct,
    sim_years,
    senior_resignations_by_year,   # list of ints, one per year
    junior_resignations_by_year,   # list of ints, one per year
    eligibility_weeks,
    seed=1
):
    random.seed(seed)
    sim_weeks = sim_years * 52
    senior_cap = int(total_team_size * senior_cap_pct)

    # ── Initialize staff ──────────────────────────────────────────────────────
    staff = []
    uid = 0

    for i in range(initial_seniors):
        staff.append({
            "id": uid, "role": "senior",
            "joined_week": -random.randint(0, 200),
            "eligible_week": None,
            "status": "active"
        })
        uid += 1

    initial_juniors = total_team_size - initial_seniors
    remaining_juniors = initial_juniors - juniors_already_eligible - juniors_eligible_in_weeks

    # Juniors already eligible for promotion
    for i in range(juniors_already_eligible):
        staff.append({
            "id": uid, "role": "junior",
            "joined_week": -eligibility_weeks,
            "eligible_week": 0,  # eligible from week 0
            "status": "active"
        })
        uid += 1

    # Juniors becoming eligible partway through simulation
    for i in range(juniors_eligible_in_weeks):
        staff.append({
            "id": uid, "role": "junior",
            "joined_week": -eligibility_weeks + 26,  # 26 weeks from eligible
            "eligible_week": 26,
            "status": "active"
        })
        uid += 1

    # Remaining juniors starting their clock now
    for i in range(max(0, remaining_juniors)):
        staff.append({
            "id": uid, "role": "junior",
            "joined_week": 0,
            "eligible_week": eligibility_weeks,
            "status": "active"
        })
        uid += 1

    # ── Generate resignation events by year ───────────────────────────────────
    def resignation_weeks_for_year(year_index, count):
        """Generate `count` random resignation weeks within the given year."""
        year_start = year_index * 52
        year_end = year_start + 52
        if count == 0:
            return []
        weeks = sorted(random.sample(range(year_start, year_end), min(count, 52)))
        return weeks

    senior_resign_weeks = []
    junior_resign_weeks = []
    for y in range(sim_years):
        s_count = senior_resignations_by_year[y] if y < len(senior_resignations_by_year) else 0
        j_count = junior_resignations_by_year[y] if y < len(junior_resignations_by_year) else 0
        senior_resign_weeks.extend(resignation_weeks_for_year(y, s_count))
        junior_resign_weeks.extend(resignation_weeks_for_year(y, j_count))

    # ── Tracking ──────────────────────────────────────────────────────────────
    history = []
    events = []
    promotion_waits = []
    open_promotion_slots = 0
    pending_junior_hires = []  # weeks when new juniors join

    # ── Week by week simulation ───────────────────────────────────────────────
    for week in range(sim_weeks):

        # Process pending junior hires becoming available
        for join_week in pending_junior_hires[:]:
            if join_week <= week:
                staff.append({
                    "id": uid, "role": "junior",
                    "joined_week": week,
                    "eligible_week": week + eligibility_weeks,
                    "status": "active"
                })
                uid += 1
                pending_junior_hires.remove(join_week)
                events.append({"week": week, "event": "New junior hire joined the team"})

        # Get current active staff
        active = [s for s in staff if s["status"] == "active"]
        seniors = [s for s in active if s["role"] == "senior"]
        juniors = [s for s in active if s["role"] == "junior"]
        eligible_juniors = sorted(
            [s for s in juniors if s["eligible_week"] is not None and s["eligible_week"] <= week],
            key=lambda s: s["eligible_week"]  # promote longest-waiting first
        )

        # Process promotions if slots are open
        while open_promotion_slots > 0 and eligible_juniors:
            candidate = eligible_juniors.pop(0)
            wait = week - candidate["eligible_week"]
            promotion_waits.append({"week": week, "wait_weeks": wait})
            candidate["role"] = "senior"
            open_promotion_slots -= 1
            events.append({"week": week, "event": f"Junior promoted to senior (waited {wait} wks after eligibility)"})

        # Senior resignations this week
        if week in senior_resign_weeks:
            active_seniors = [s for s in staff if s["status"] == "active" and s["role"] == "senior"]
            if active_seniors:
                leaver = random.choice(active_seniors)
                leaver["status"] = "resigned"
                events.append({"week": week, "event": f"Senior resigned (week {week}, year {week//52+1})"})

                # Always replace with junior to maintain team size
                pending_junior_hires.append(week)  # available immediately (no onboarding delay in pipeline)
                events.append({"week": week, "event": "Junior replacement hired"})

                # Check if now below senior cap
                current_seniors = len([s for s in staff if s["status"] == "active" and s["role"] == "senior"])
                current_total = len([s for s in staff if s["status"] == "active"])
                cap = int(current_total * senior_cap_pct)
                if current_seniors <= cap:
                    open_promotion_slots += 1
                    events.append({"week": week, "event": "⭐ Promotion slot opened (below senior cap)"})
                else:
                    events.append({"week": week, "event": "No promotion slot (still at/above senior cap)"})

        # Junior resignations this week
        if week in junior_resign_weeks:
            active_juniors = [s for s in staff if s["status"] == "active" and s["role"] == "junior"]
            if active_juniors:
                # Prefer to remove a non-eligible junior to preserve pipeline
                non_eligible = [s for s in active_juniors if s["eligible_week"] is None or s["eligible_week"] > week]
                leaver = random.choice(non_eligible if non_eligible else active_juniors)
                leaver["status"] = "resigned"
                events.append({"week": week, "event": f"Junior resigned (week {week}, year {week//52+1}) — eligibility progress lost"})
                # Replacement starts eligibility clock from scratch
                pending_junior_hires.append(week)


        # Weekly snapshot
        active = [s for s in staff if s["status"] == "active"]
        seniors_now = len([s for s in active if s["role"] == "senior"])
        juniors_now = len([s for s in active if s["role"] == "junior"])
        eligible_now = len([s for s in active if s["role"] == "junior"
                           and s["eligible_week"] is not None
                           and s["eligible_week"] <= week])
        fixed_cap = int(total_team_size * senior_cap_pct)  # fixed baseline

        history.append({
            "week": week,
            "year": week // 52 + 1,
            "week_of_year": week % 52 + 1,
            "seniors": seniors_now,
            "juniors": juniors_now,
            "total": seniors_now + juniors_now,
            "eligible_juniors": eligible_now,
            "open_slots": open_promotion_slots,
            "senior_pct": seniors_now / max(seniors_now + juniors_now, 1) * 100,
            "senior_cap": fixed_cap,
        })

    return pd.DataFrame(history), events, promotion_waits, senior_cap


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.header("Simulation Controls")

st.sidebar.subheader("👥 Initial Team")
total_team_size = st.sidebar.number_input("Total Team Size", 5, 50, 14)
initial_seniors = st.sidebar.number_input("Initial Seniors", 1, int(total_team_size), 11)
st.sidebar.caption(f"Initial juniors: {int(total_team_size) - int(initial_seniors)}")
senior_cap_pct  = st.sidebar.slider("Senior Cap (% of team)", 10, 90, 50) / 100
st.sidebar.caption(f"Senior cap = {int(int(total_team_size) * senior_cap_pct)} staff at current team size")

st.sidebar.subheader("📅 Simulation Length")
sim_years = st.sidebar.slider("Simulation Length (years)", 1, 5, 3)

st.sidebar.subheader("📉 Senior Resignations by Year")
senior_res = []
for y in range(sim_years):
    val = st.sidebar.number_input(f"Year {y+1}", 0, 10, 2 if y == 0 else (1 if y == 1 else 0),
                                   key=f"senior_res_{y}")
    senior_res.append(int(val))

st.sidebar.subheader("📉 Junior Resignations by Year")
junior_res = []
for y in range(sim_years):
    val = st.sidebar.number_input(f"Year {y+1} ", 0, 10, 0,
                                   key=f"junior_res_{y}")
    junior_res.append(int(val))

st.sidebar.subheader("🎯 Current Junior Pipeline")
juniors_already_eligible = st.sidebar.number_input("Juniors already eligible for promotion", 0, 10, 2)
juniors_eligible_in_weeks_input = st.sidebar.number_input("Juniors becoming eligible in ~6 months", 0, 10, 1)
st.sidebar.caption(f"Remaining juniors start their eligibility clock from week 0")

st.sidebar.subheader("⚙️ Promotion Settings")
eligibility_weeks = st.sidebar.slider("Promotion Eligibility (weeks in org)", 26, 130, 78,
                                       help="78 weeks = 18 months")

st.sidebar.subheader("🎲 Randomization")
seed = st.sidebar.slider("Seed", 1, 100, 1)

# ── Run simulation ────────────────────────────────────────────────────────────

df, events, promotion_waits, senior_cap = run_simulation(
    total_team_size=int(total_team_size),
    initial_seniors=int(initial_seniors),
    senior_cap_pct=senior_cap_pct,
    sim_years=sim_years,
    senior_resignations_by_year=senior_res,
    junior_resignations_by_year=junior_res,
    juniors_already_eligible=int(juniors_already_eligible),
    juniors_eligible_in_weeks=int(juniors_eligible_in_weeks_input),
    eligibility_weeks=eligibility_weeks,
    seed=seed
)

# ── Summary metrics ───────────────────────────────────────────────────────────

st.subheader("Simulation Summary")

total_senior_res = sum(senior_res)
total_junior_res = sum(junior_res)
avg_wait = int(sum(p["wait_weeks"] for p in promotion_waits) / max(len(promotion_waits), 1)) if promotion_waits else 0

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Senior Cap", f"{int(senior_cap_pct*100)}% = {senior_cap} staff")
col2.metric("Senior Resignations", total_senior_res)
col3.metric("Junior Resignations", total_junior_res)
col4.metric("Total Promotions", len(promotion_waits))
col5.metric("Avg Wait After Eligibility", f"{avg_wait} weeks" if promotion_waits else "No promotions")

# Alerts
final = df.iloc[-1]
if final["open_slots"] > 0 and final["eligible_juniors"] == 0:
    st.error("🚨 Critical pipeline gap: open promotion slots exist but NO eligible juniors available!")
elif final["open_slots"] > 0 and final["eligible_juniors"] > 0:
    st.warning(f"⚠️ {int(final['open_slots'])} open promotion slot(s) with {int(final['eligible_juniors'])} eligible junior(s) waiting.")
elif final["senior_pct"] > senior_cap_pct * 100:
    st.warning(f"⚠️ Team ends simulation above the senior cap ({final['senior_pct']:.1f}% vs {int(senior_cap_pct*100)}% cap).")
else:
    st.success("✅ Team ends simulation within cap with no critical pipeline gaps.")

st.markdown("---")

# ── Chart 1: Team composition ─────────────────────────────────────────────────

st.subheader("Team Composition Over Time")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df["week"], y=df["seniors"],
    name="Seniors", mode="lines",
    line=dict(color="#c9a84c", width=2),
    fill="tozeroy", fillcolor="rgba(201,168,76,0.15)"
))
fig1.add_trace(go.Scatter(
    x=df["week"], y=df["juniors"],
    name="Juniors", mode="lines",
    line=dict(color="#7eb8c9", width=2),
))
fig1.add_trace(go.Scatter(
    x=df["week"], y=df["eligible_juniors"],
    name="Eligible for Promotion", mode="lines",
    line=dict(color="#9b7eb8", width=2, dash="dot"),
))
fig1.add_trace(go.Scatter(
    x=df["week"], y=df["open_slots"],
    name="Open Promotion Slots", mode="lines",
    line=dict(color="#e8734a", width=2, dash="dash"),
))
fig1.add_trace(go.Scatter(
    x=df["week"], y=df["senior_cap"],
    name="Senior Cap", mode="lines",
    line=dict(color="#c9a84c", width=1, dash="dot"),
))

for y in range(1, sim_years + 1):
    fig1.add_vline(x=y * 52, line_width=1, line_color="#2a2a3a",
                   annotation_text=f"Year {y}", annotation_position="top")

fig1.update_layout(
    paper_bgcolor="#0f0f14", plot_bgcolor="#0f0f14",
    font=dict(color="#e8e4d9", family="Georgia"),
    xaxis=dict(title="Week", gridcolor="#2a2a3a", tickfont=dict(size=10)),
    yaxis=dict(title="Number of Staff", gridcolor="#2a2a3a"),
    legend=dict(bgcolor="#1a1a24", bordercolor="#2a2a3a"),
    height=420, margin=dict(l=60, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor="#1a1a24"),
)
st.plotly_chart(fig1, use_container_width=True)

# ── Chart 2: Senior % over time ───────────────────────────────────────────────

st.subheader("Senior % of Team Over Time")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df["week"], y=df["senior_pct"],
    name="Senior %", mode="lines",
    line=dict(color="#c9a84c", width=2),
    fill="tozeroy", fillcolor="rgba(201,168,76,0.1)"
))
fig2.add_hline(y=senior_cap_pct * 100, line_dash="dot", line_color="#e8734a",
               annotation_text=f"Cap ({int(senior_cap_pct*100)}%)",
               annotation_position="top right")

for y in range(1, sim_years + 1):
    fig2.add_vline(x=y * 52, line_width=1, line_color="#2a2a3a")

fig2.update_layout(
    paper_bgcolor="#0f0f14", plot_bgcolor="#0f0f14",
    font=dict(color="#e8e4d9", family="Georgia"),
    xaxis=dict(title="Week", gridcolor="#2a2a3a"),
    yaxis=dict(title="% Senior", gridcolor="#2a2a3a", range=[0, 100]),
    height=300, margin=dict(l=60, r=20, t=40, b=40),
    hoverlabel=dict(bgcolor="#1a1a24"),
)
st.plotly_chart(fig2, use_container_width=True)

# ── Promotion summary callout ────────────────────────────────────────────────

st.subheader("Promotion Pipeline Summary")

if promotion_waits:
    total_promos = len(promotion_waits)
    avg_wait = int(sum(p["wait_weeks"] for p in promotion_waits) / total_promos)
    max_wait = max(p["wait_weeks"] for p in promotion_waits)
    min_wait = min(p["wait_weeks"] for p in promotion_waits)
    
    pcol1, pcol2, pcol3 = st.columns(3)
    pcol1.metric("Total Promotions", total_promos)
    pcol2.metric("Avg Wait After Eligibility", f"{avg_wait} weeks")
    pcol3.metric("Longest Wait", f"{max_wait} weeks")

    if avg_wait == 0:
        wait_msg = "Promoted staff were promoted immediately upon eligibility — slots opened fast enough to keep up with the pipeline."
    elif avg_wait < 13:
        wait_msg = f"Of {total_promos} promotion(s) during this simulation, staff waited an average of {avg_wait} weeks after becoming eligible before a slot opened. Waits ranged from {min_wait} to {max_wait} weeks."
    else:
        wait_msg = f"Of {total_promos} promotion(s) during this simulation, staff waited an average of {avg_wait} weeks after becoming eligible — over 3 months. This suggests attrition is too slow to keep pace with junior eligibility. Waits ranged from {min_wait} to {max_wait} weeks."
    
    st.info(wait_msg)
else:
    st.warning("No promotions occurred during this simulation. This may mean attrition is too slow to open slots, or no juniors reached eligibility in time.")

# ── Event log ─────────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Event Log")

with st.expander("Show full event log"):
    event_df = pd.DataFrame(events)
    event_df["Year"] = (event_df["week"] // 52) + 1
    event_df["Week of Year"] = event_df["week"] % 52 + 1
    event_df = event_df.rename(columns={"week": "Overall Week", "event": "Event"})
    event_df = event_df[["Year", "Week of Year", "Overall Week", "Event"]]
    st.dataframe(event_df, use_container_width=True)
