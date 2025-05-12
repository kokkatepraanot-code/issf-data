# app.py
# test
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
from io import BytesIO


# === Simple Login Credentials ===
USER_CREDENTIALS = {
    "admin": "gojags2025"
}

# === Login Check ===
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Login Required")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_btn = st.button("Login")

    if login_btn:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.success("✅ Login successful. Loading dashboard...")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

    st.stop()  # Stop rendering the rest of the app until authenticated


# Remove top margin
st.markdown("""
    <style>
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

# Add logo and title
st.sidebar.image("logo.png", width=250)
st.sidebar.markdown("### The International School of San Francisco")
st.sidebar.markdown("---")
st.markdown("### IB Diploma Programme - Performance Dashboard")
st.markdown("---")


# Load EE/TOK data
@st.cache_data
def load_ee_tok_data():
    df = pd.read_csv("ee_tok_points_df.csv")
    df[['EE_Grade', 'TOK_Grade']] = df['EE/TOK Grade'].apply(
        lambda x: pd.Series(list(x)) if isinstance(x, str) and len(x) == 2 else pd.Series([None, None])
    )
    grade_to_points = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
    df['EE_Numeric'] = df['EE_Grade'].map(grade_to_points)
    df['TOK_Numeric'] = df['TOK_Grade'].map(grade_to_points)
    return df

# Load DP subject data (used for all analysis)
@st.cache_data
def load_dp_data():
    df = pd.read_csv("student_points_df.csv")
    df["Subject grade"] = pd.to_numeric(df["Subject grade"], errors="coerce")
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    return df

# Load datasets
df = load_ee_tok_data()
df1 = load_dp_data()

# === Add IB Group Mapping ===
group_mapping = {
    # Group 1: Studies in Language and Literature
    "ENGLISH A: Language and Literature": 1,
    "ENGLISH A: Literature": 1,
    "ENGLISH A LIT (P)": 1,
    "SPANISH A: Language and Literature": 1,
    "FRENCH A: Language and Literature": 1,
    "FRENCH A: Literature": 1,
    "CHINESE A: Language and Literature": 1,
    "CHINESE A: Literature": 1,
    "GERMAN A: Language and Literature": 1,
    "GERMAN A: Literature": 1,
    "ITALIAN A: Literature": 1,
    "DUTCH A: Literature": 1,
    "NORWEGIAN A: Literature": 1,
    "PORTUGUESE A: Literature": 1,
    "VIETNAMESE A: Literature": 1,

    # Group 2: Language Acquisition
    "FRENCH B": 2,
    "SPANISH B": 2,
    "GERMAN B": 2,
    "ITALIAN B": 2,
    "ENGLISH B": 2,
    "CHINESE B": 2,
    "ARABIC B": 2,
    "FRENCH AB INITIO.": 2,
    "SPANISH AB INITIO.": 2,
    "ARABIC AB INITIO.": 2,
    "MANDARIN AB INITIO.": 2,

    # Group 3: Individuals and Societies
    "ECONOMICS": 3,
    "ECONOMICS (P)": 3,
    "BUSINESS AND MANAGEMENT": 3,
    "BUS MAN": 3,
    "HISTORY": 3,
    "GEOGRAPHY": 3,
    "PSYCHOLOGY": 3,
    "PSYCHOLOGY (P)": 3,
    "PHILOSOPHY": 3,
    "GLOB. POL": 3,
    "INFORMATION TECHNOLOGY IN GLOBAL SOCIETY": 3,

    # Group 4: Sciences
    "BIOLOGY": 4,
    "CHEMISTRY": 4,
    "PHYSICS": 4,
    "ENVIRONMENTAL SYSTEMS AND SOCIETIES": 4,

    # Group 5: Mathematics
    "MATH ANALYSIS": 5,
    "MATH APPS": 5,
    "MATH.STUDIES": 5,
    "MATHEMATICS": 5,

    # Group 6: The Arts
    "VISUAL ARTS": 6,
    "THEATRE": 6,
    "MUSIC": 6,
    "FILM": 6,
    "DESIGN TECHNOLOGY": 6,
    "COMPUTER SCIENCE": 6
}

# Apply group mapping
df1["Group"] = df1["Subject"].map(group_mapping)


views_with_year = [
    "EE Grade Distribution (%)",
    "TOK Grade Distribution (%)"
]

all_views = [
    "Average Grades Trend",
    "EE/TOK Average Points Trend",
    "EE Grade Distribution (%)",
    "TOK Grade Distribution (%)",
    "EE/TOK Points Distribution (Counts)",
    "EE Grade Distribution (Counts)",
    "TOK Grade Distribution (Counts)",
    "Per-Year Grade & Point Breakdown"
]

tab1, tab2 = st.tabs(["EE/TOK Analysis", "IB Diploma Summary"])

with tab1:
    st.sidebar.markdown("## Filters")
    view = st.sidebar.selectbox("Select a View", all_views)
    year = st.sidebar.selectbox("Select Year", sorted(df["Year"].unique())) if view in views_with_year else None

    if view != "Per-Year Grade & Point Breakdown":
        fig, ax = plt.subplots(figsize=(9, 4))

        if view == "Average Grades Trend":
            avg_df = df.groupby("Year")[["EE_Numeric", "TOK_Numeric", "Numeric EE/TOK Points"]].mean().reset_index()
            ax.plot(avg_df["Year"], avg_df["EE_Numeric"], marker='o', label="EE Avg")
            ax.plot(avg_df["Year"], avg_df["TOK_Numeric"], marker='s', label="TOK Avg")
            ax.plot(avg_df["Year"], avg_df["Numeric EE/TOK Points"], marker='^', label="Combined")
            ax.set_title("EE & TOK Average Grade Trends")

        elif view == "EE/TOK Average Points Trend":
            trend_df = df.groupby("Year")["Numeric EE/TOK Points"].mean().reset_index()
            ax.plot(trend_df["Year"], trend_df["Numeric EE/TOK Points"], marker='o', color='orange')
            ax.set_title("EE/TOK Average Points Trend")

        elif view == "EE Grade Distribution (%)":
            sub = df[df["Year"] == year]
            dist = sub["EE_Grade"].value_counts(normalize=True).sort_index() * 100
            dist.plot(kind='bar', color='mediumseagreen', ax=ax)
            ax.set_title(f"EE Grade Distribution in {year} (%)")
            ax.set_ylim(0, 100)

        elif view == "TOK Grade Distribution (%)":
            sub = df[df["Year"] == year]
            dist = sub["TOK_Grade"].value_counts(normalize=True).sort_index() * 100
            dist.plot(kind='bar', color='cornflowerblue', ax=ax)
            ax.set_title(f"TOK Grade Distribution in {year} (%)")
            ax.set_ylim(0, 100)

        elif view == "EE/TOK Points Distribution (Counts)":
            dist = df.groupby(['Year', 'Numeric EE/TOK Points']).size().unstack().fillna(0)
            dist.plot(kind='bar', stacked=True, colormap='viridis', ax=ax)
            ax.set_title("EE/TOK Points Distribution by Year")

        elif view == "EE Grade Distribution (Counts)":
            dist = df.groupby(['Year', 'EE_Grade']).size().unstack().fillna(0)
            dist.plot(kind='bar', stacked=True, colormap='copper', ax=ax)
            ax.set_title("EE Grade Distribution by Year")

        elif view == "TOK Grade Distribution (Counts)":
            dist = df.groupby(['Year', 'TOK_Grade']).size().unstack().fillna(0)
            dist.plot(kind='bar', stacked=True, colormap='plasma', ax=ax)
            ax.set_title("TOK Grade Distribution by Year")

        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

    else:
        st.subheader("Per-Year Grade & Point Breakdown")
        for yr in sorted(df['Year'].unique()):
            st.markdown(f"### Year: {yr}")
            col1, col2, col3 = st.columns(3)
            year_df = df[df['Year'] == yr]

            with col1:
                grade_counts = year_df['EE_Grade'].value_counts().sort_index()
                fig, ax = plt.subplots()
                sns.barplot(x=grade_counts.index, y=grade_counts.values, ax=ax, palette="Blues_d")
                ax.set_title("EE Grades")
                ax.set_ylabel("Count")
                ax.set_xlabel("Grade")
                st.pyplot(fig)

            with col2:
                tok_counts = year_df['TOK_Grade'].value_counts().sort_index()
                fig, ax = plt.subplots()
                sns.barplot(x=tok_counts.index, y=tok_counts.values, ax=ax, palette="Greens_d")
                ax.set_title("TOK Grades")
                ax.set_ylabel("Count")
                ax.set_xlabel("Grade")
                st.pyplot(fig)

            with col3:
                points_counts = year_df['Numeric EE/TOK Points'].value_counts().sort_index()
                fig, ax = plt.subplots()
                sns.barplot(x=points_counts.index.astype(str), y=points_counts.values, ax=ax, palette="Purples_d")
                ax.set_title("EE + TOK Total Points")
                ax.set_ylabel("Count")
                ax.set_xlabel("Points")
                st.pyplot(fig)

with tab2:
    st.sidebar.markdown("## 📈 IB Trends")
    
    ib_trends = [
    "📋 Annual Breakdown",
    "📊 Multi-Year Summary Trends",
    "Average Subject Grade by Level",
    "Top 5 Most Popular Subjects",
    "Subject Grade Trend Over Time",
    "EE/TOK Bonus vs Total DP Points Correlation",
    "🔢 % of Students Scoring 6 or 7 Over Time",
    "📈 Average Total Points Over Time",
    "🏆 Subject Performance Over Time",
    "🎓 Pass Rate Trends for DIPLOMA Candidates",
    "📚 IB Group-Wise Subject Trends",
    "📤 Compare School vs World Averages (Upload PDFs)"

]
    
    selected_trend = st.sidebar.selectbox("Select IB Diploma Trend", ib_trends)

    if selected_trend == "Average Subject Grade by Level":
        st.markdown("### 📊 Average Subject Grade by Level Over Time")
        avg_by_level = df1.groupby(["Year", "Level"])["Subject grade"].mean().unstack()
        fig, ax = plt.subplots(figsize=(8, 4))
        avg_by_level.plot(marker='o', ax=ax)
        ax.set_title("Average Subject Grade: HL vs SL")
        ax.set_ylabel("Average Grade")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 7)
        ax.grid(True)
        st.pyplot(fig)

    elif selected_trend == "📋 Annual Breakdown":
        st.subheader("Annual IB DP Summary")
        years_available = sorted(df1["Year"].dropna().unique())
        selected_year = st.slider(
            "Select Year for Detailed Summary",
            min_value=int(min(years_available)),
            max_value=int(max(years_available)),
            value=int(max(years_available)),
            step=1
        )


        if selected_year:
            year_df = df1[df1['Year'] == selected_year]

            total_students = year_df['Name'].nunique()
            levels = year_df['Level'].dropna().unique()
            categories = year_df['Category'].dropna().unique()
            subjects = year_df['Subject'].nunique()
            languages = year_df['Language'].nunique()

            diploma_df = year_df[year_df['Category'] == 'DIPLOMA']
            diploma_students = diploma_df['Name'].nunique()
            course_students = year_df[year_df['Category'] == 'COURSE']['Name'].nunique()
            retake_students = year_df[year_df['Category'] == 'RETAKE']['Name'].nunique()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"- **Total students:** {total_students}")
                st.markdown(f"- **Levels ({len(levels)}):** {list(levels)}")
                st.markdown(f"- **Categories ({len(categories)}):** {list(categories)}")
                st.markdown(f"- **Subjects:** {subjects}")
                st.markdown(f"- **Languages:** {languages}")
                st.markdown(f"- **Diploma students:** {diploma_students}")
                st.markdown(f"- **Course students:** {course_students}")
                st.markdown(f"- **Retake students:** {retake_students}")

            with col2:
                st.subheader("Further Statistics for Diploma Students")

                merged_df = pd.merge(
                    diploma_df,
                    df[['Name', 'Year', 'Numeric EE/TOK Points']],
                    on=['Name', 'Year'],
                    how='left'
                )

                subject_scores = merged_df.dropna(subset=["Subject grade"])
                subject_totals = subject_scores.groupby("Name")["Subject grade"].agg(['sum', 'count'])
                bonus_points = merged_df.groupby("Name")["Numeric EE/TOK Points"].first()

                combined_totals = subject_totals[subject_totals['count'] >= 6]['sum'] + bonus_points
                valid_totals = combined_totals.dropna()

                if not valid_totals.empty:
                    st.markdown(f"- **Mean Total Point:** {valid_totals.mean():.1f}")
                    st.markdown(f"- **Highest Total Point:** {valid_totals.max():.1f}")
                    st.markdown(f"- **75% of students achieved:** {valid_totals.quantile(0.75):.1f} points")
                    st.markdown(f"- **50% of students achieved:** {valid_totals.median():.1f} points")

                    st.markdown("### Distribution of Total Points")
                    fig, ax = plt.subplots(figsize=(6, 3))
                    sns.histplot(valid_totals, bins=10, kde=True, color='skyblue', ax=ax)
                    ax.set_title("Histogram of Total Diploma Points")
                    ax.set_xlabel("Total Points")
                    ax.set_ylabel("Number of Students")
                    st.pyplot(fig)

                    percentile = st.slider("Select Percentile", 0, 100, 75)
                    percentile_value = valid_totals.quantile(percentile / 100)
                    st.markdown(f"**{percentile}% of students achieved:** {percentile_value:.1f} points")

    elif selected_trend == "📊 Multi-Year Summary Trends":
        st.markdown("### 📊 Multi-Year Diploma Summary Trends")

        # Compute yearly statistics
        yearly_summary = []
        for year in sorted(df1["Year"].dropna().unique()):
            year_df = df1[df1['Year'] == year]
            diploma_df = year_df[year_df["Category"] == "DIPLOMA"]

            merged_df = pd.merge(
                diploma_df,
                df[["Name", "Year", "Numeric EE/TOK Points"]],
                on=["Name", "Year"],
                how="left"
            )

            subject_scores = merged_df.dropna(subset=["Subject grade"])
            subject_totals = subject_scores.groupby("Name")["Subject grade"].agg(['sum', 'count'])
            bonus_points = merged_df.groupby("Name")["Numeric EE/TOK Points"].first()

            combined_totals = subject_totals[subject_totals['count'] >= 6]['sum'] + bonus_points
            valid_totals = combined_totals.dropna()

            if not valid_totals.empty:
                yearly_summary.append({
                    "Year": year,
                    "Total Students": year_df["Name"].nunique(),
                    "Diploma Students": diploma_df["Name"].nunique(),
                    "Course Students": year_df[year_df["Category"] == "COURSE"]["Name"].nunique(),
                    "Mean Total Point": valid_totals.mean(),
                    "Highest Total Point": valid_totals.max(),
                    "75th Percentile": valid_totals.quantile(0.75),
                    "50th Percentile": valid_totals.median()
                })

        summary_df = pd.DataFrame(yearly_summary)

        if not summary_df.empty:
            # Plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Student Counts
            axes[0, 0].plot(summary_df["Year"], summary_df["Total Students"], marker='o', label="Total")
            axes[0, 0].plot(summary_df["Year"], summary_df["Diploma Students"], marker='o', label="Diploma")
            axes[0, 0].plot(summary_df["Year"], summary_df["Course Students"], marker='o', label="Course")
            axes[0, 0].set_title("Student Counts by Year")
            axes[0, 0].legend()
            axes[0, 0].set_ylabel("Students")
            axes[0, 0].set_xlabel("Year")

            # Mean vs Max Points
            axes[0, 1].plot(summary_df["Year"], summary_df["Mean Total Point"], marker='o', label="Mean")
            axes[0, 1].plot(summary_df["Year"], summary_df["Highest Total Point"], marker='o', label="Max")
            axes[0, 1].set_title("Mean and Highest Total Points")
            axes[0, 1].legend()
            axes[0, 1].set_ylabel("Points")
            axes[0, 1].set_xlabel("Year")

            # Percentiles
            axes[1, 0].plot(summary_df["Year"], summary_df["75th Percentile"], marker='o', label="75th Percentile")
            axes[1, 0].plot(summary_df["Year"], summary_df["50th Percentile"], marker='o', label="Median")
            axes[1, 0].set_title("Performance Percentiles")
            axes[1, 0].legend()
            axes[1, 0].set_ylabel("Points")
            axes[1, 0].set_xlabel("Year")

            # Hide empty subplot
            axes[1, 1].axis("off")

            fig.tight_layout()
            st.pyplot(fig)

            with st.expander("📋 View Data Table"):
                st.dataframe(summary_df.set_index("Year").style.format("{:.1f}", subset=["Mean Total Point", "Highest Total Point", "75th Percentile", "50th Percentile"]))

        else:
            st.warning("No valid data available to display the multi-year trends.")

    
    elif selected_trend == "Top 5 Most Popular Subjects":
        st.markdown("### 🏆 Top 5 Most Popular Subjects (by Student Count)")
        selected_year = st.selectbox("Select Year", sorted(df1["Year"].dropna().unique()), key="popular_year")

        top_subjects = (
            df1[df1["Year"] == selected_year]
            .groupby("Subject")["Name"]
            .nunique()
            .sort_values(ascending=False)
            .head(5)
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=top_subjects.values, y=top_subjects.index, palette="viridis", ax=ax)
        ax.set_title(f"Top 5 Subjects in {selected_year}")
        ax.set_xlabel("Number of Students")
        ax.set_ylabel("Subject")
        st.pyplot(fig)
    
    elif selected_trend == "Subject Grade Trend Over Time":
        st.markdown("### 📈 Subject Grade Trend Over Time")
        available_subjects = df1["Subject"].dropna().unique()
        selected_subject = st.selectbox("Select Subject", sorted(available_subjects))

        trend_df = df1[df1["Subject"] == selected_subject]
        trend_df["Year"] = trend_df["Year"].astype(int)
        yearly_avg = trend_df.groupby("Year")["Subject grade"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.lineplot(data=yearly_avg, x="Year", y="Subject grade", marker="o", ax=ax)
        ax.set_xticks(sorted(yearly_avg["Year"].unique()))  # Force integer ticks only
        ax.set_title(f"Average Grade Over Time: {selected_subject}")
        ax.set_ylabel("Average Grade")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 7)
        ax.grid(True)
        st.pyplot(fig)

    elif selected_trend == "EE/TOK Bonus vs Total DP Points Correlation":
        st.markdown("### 🧠 EE/TOK Bonus Points vs Total Diploma Score")

        diploma_df = df1[df1['Category'] == 'DIPLOMA']
        merged_df = pd.merge(diploma_df, df[['Name', 'Year', 'Numeric EE/TOK Points']], on=['Name', 'Year'], how='left')

        subject_scores = merged_df.dropna(subset=["Subject grade"])
        subject_totals = subject_scores.groupby("Name")["Subject grade"].agg(['sum', 'count'])
        bonus_points = merged_df.groupby("Name")["Numeric EE/TOK Points"].first()

        # Filter students with 6 or more subjects
        valid_names = subject_totals[subject_totals['count'] >= 6].index
        bonus_values = bonus_points.loc[bonus_points.index.isin(valid_names)]
        total_points = subject_totals.loc[valid_names]['sum'] + bonus_values

        valid_df = pd.DataFrame({
            "Total Points": total_points,
            "Bonus Points": bonus_values
        }).dropna()

        if not valid_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.scatterplot(data=valid_df, x="Bonus Points", y="Total Points", ax=ax)
            sns.regplot(data=valid_df, x="Bonus Points", y="Total Points", scatter=False, ax=ax, color='red', line_kws={"linestyle":"--"})
            ax.set_title("Correlation between EE/TOK Bonus and Total Diploma Score")
            ax.set_xlabel("EE/TOK Bonus Points")
            ax.set_ylabel("Total Diploma Points")
            st.pyplot(fig)

            corr = valid_df["Bonus Points"].corr(valid_df["Total Points"])
            st.markdown(f"**Pearson Correlation Coefficient:** `{corr:.2f}`")
            st.markdown("""
            **🧾 Interpretation Note:**

            This plot shows a **positive correlation** between students' EE/TOK bonus points and their total IB Diploma scores. Each dot represents a student, and the red dashed line shows the general trend.

            - A correlation coefficient (r) closer to **1** means a **strong positive** relationship.
            - In this case, students who earn more EE/TOK points also tend to achieve **higher overall diploma scores**.
            - This suggests the importance of EE/TOK performance in predicting overall success.

            **Insight:** Emphasize support for EE/TOK preparation, as it likely contributes to broader academic performance.
            """)

        else:
            st.warning("No valid data to display. Make sure students have at least 6 subject grades and EE/TOK points.")

    elif selected_trend == "📈 Average Total Points Over Time":
        st.markdown("### 📈 Average Total Points Over Time")

        # Merge to get bonus points per student
        merged_df = pd.merge(df1, df[['Name', 'Year', 'Numeric EE/TOK Points']], on=['Name', 'Year'], how='left')
        diploma_df = merged_df[merged_df["Category"] == "DIPLOMA"]

        # Calculate total score only for students with 6+ subjects
        subject_scores = diploma_df.dropna(subset=["Subject grade"])
        subject_totals = subject_scores.groupby(["Year", "Name"])["Subject grade"].agg(['sum', 'count'])
        bonus_points = diploma_df.groupby(["Year", "Name"])["Numeric EE/TOK Points"].first()

        combined = subject_totals[subject_totals['count'] >= 6]['sum'] + bonus_points
        avg_totals_by_year = combined.groupby("Year").mean()

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        avg_totals_by_year.plot(marker='o', color='dodgerblue', ax=ax)
        ax.set_title("Average Total Diploma Points by Year")
        ax.set_ylabel("Average Total Points")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 45)
        ax.grid(True)
        for x, y in avg_totals_by_year.items():
            ax.text(x, y + 0.5, f"{y:.1f}", ha='center', fontsize=9)
        st.pyplot(fig)


    elif selected_trend == "🔢 % of Students Scoring 6 or 7 Over Time":
        st.markdown("### 🔢 % of Students Scoring 6 or 7 Over Time by Level (HL vs SL)")

        diploma_only = df1[df1["Category"] == "DIPLOMA"]
        diploma_only = diploma_only[diploma_only["Subject grade"].notna()]

        # Compute percentage of students scoring 6 or 7 for each Year & Level
        summary = (
            diploma_only.assign(high_score=(diploma_only["Subject grade"] >= 6).astype(int))
            .groupby(["Year", "Level"])["high_score"]
            .agg(["sum", "count"])
            .reset_index()
        )
        summary["% High Scores"] = (summary["sum"] / summary["count"]) * 100

        # Pivot for plotting
        pivot = summary.pivot(index="Year", columns="Level", values="% High Scores").sort_index()

        # Plot using matplotlib and annotate with tooltips
        fig, ax = plt.subplots(figsize=(8, 4))
        for level in pivot.columns:
            ax.plot(pivot.index, pivot[level], marker='o', label=level)

            # for x, y in zip(pivot.index, pivot[level]):
            #     ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=8)

        ax.set_title("% of Students Scoring 6 or 7 by Level")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 100)
        ax.legend(title="Level")
        ax.grid(True)
        st.pyplot(fig)

    elif selected_trend == "🏆 Subject Performance Over Time":
        # Sidebar section: Subject Performance Filters
        st.sidebar.markdown("## 🎯 Subject Performance Explorer")
        min_students = st.sidebar.slider("Minimum number of students per subject", 1, 30, 5)
        year_range = st.sidebar.slider("Select year range", int(df1["Year"].min()), int(df1["Year"].max()), (2014, 2024))
        level_options = st.sidebar.multiselect("Select Levels", ["HL", "SL"], default=["HL", "SL"])

        st.markdown("### 🏆 Subject Performance Over Time")
        # Filter DP dataset
        filtered_df = df1[
            (df1["Year"] >= year_range[0]) &
            (df1["Year"] <= year_range[1]) &
            (df1["Level"].isin(level_options))
        ]

        # Count and average subject grades
        subject_counts = filtered_df.groupby(["Subject", "Year"]).size().reset_index(name="count")
        avg_grades = filtered_df.groupby(["Subject", "Year"])["Subject grade"].mean().reset_index()

        # Merge and filter by min student count
        merged_subjects = pd.merge(avg_grades, subject_counts, on=["Subject", "Year"])
        valid_subjects = merged_subjects[merged_subjects["count"] >= min_students]


        # Identify top 5 and bottom 5 subjects by overall average
        mean_by_subject = valid_subjects.groupby("Subject")["Subject grade"].mean().reset_index()
        top5 = mean_by_subject.sort_values("Subject grade", ascending=False).head(5)["Subject"]
        bottom5 = mean_by_subject.sort_values("Subject grade", ascending=True).head(5)["Subject"]

        top_data = valid_subjects[valid_subjects["Subject"].isin(top5)]
        bottom_data = valid_subjects[valid_subjects["Subject"].isin(bottom5)]


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        for subject, data in top_data.groupby("Subject"):
            ax1.plot(data["Year"], data["Subject grade"], marker='o', label=subject)
        ax1.set_title("Top Performing Subjects")
        ax1.set_ylabel("Subject grade")
        ax1.set_xlabel("Year")
        ax1.set_xticks(sorted(top_data["Year"].unique()))  # ✅ FIX
        ax1.legend()

        for subject, data in bottom_data.groupby("Subject"):
            ax2.plot(data["Year"], data["Subject grade"], marker='o', label=subject)
            
        ax2.set_title("Lowest Performing Subjects")
        ax2.set_xlabel("Year")
        ax2.set_xticks(sorted(bottom_data["Year"].unique()))  # ✅ FIX
        ax2.legend()
        st.pyplot(fig)


    elif selected_trend == "🎓 Pass Rate Trends for DIPLOMA Candidates":
        st.markdown("### 🎓 Pass Rate Trends for DIPLOMA Candidates Over Time")

        diploma_df = df1[df1["Category"] == "DIPLOMA"]
        merged = pd.merge(diploma_df, df[["Name", "Year", "Numeric EE/TOK Points"]], on=["Name", "Year"], how="left")
        merged = merged.dropna(subset=["Subject grade"])

        pass_summary = []

        for year in sorted(merged["Year"].dropna().unique()):
            year_data = merged[merged["Year"] == year]
            students = year_data.groupby("Name")

            total = 0
            passed = 0

            for name, group in students:
                grades = group["Subject grade"].dropna().tolist()
                bonus = group["Numeric EE/TOK Points"].iloc[0]

                if len(grades) >= 6:
                    total += 1
                    total_score = sum(sorted(grades, reverse=True)[:6]) + bonus
                    num_1s = grades.count(1)
                    num_2s = grades.count(2)
                    num_3_or_less = sum(1 for g in grades if g <= 3)

                    # Standard IB DP rules
                    if (
                        total_score >= 24 and
                        num_1s == 0 and
                        num_2s <= 1 and
                        num_3_or_less <= 3
                    ):
                        passed += 1

            if total > 0:
                pass_summary.append({"Year": year, "Total": total, "Passed": passed, "Pass %": (passed / total) * 100})

        pass_rates = pd.DataFrame(pass_summary)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))

        # Plot line
        ax.plot(pass_rates["Year"], pass_rates["Pass %"], marker='o', color='seagreen')

        # Titles and axis labels
        ax.set_title("📈 Pass Rate for Diploma Candidates", fontsize=16, pad=20)
        ax.set_ylabel("Pass %", fontsize=12)
        ax.set_xlabel("Year", fontsize=12)

        # Set Y-axis range with headroom
        ax.set_ylim(70, 105)
        ax.set_yticks(range(70, 111, 5))
        ax.grid(True, linestyle='--', alpha=0.5)

        # Add value labels above the points
        for i, row in pass_rates.iterrows():
            ax.text(row["Year"], row["Pass %"] + 1.5, f"{row['Pass %']:.1f}%", 
                    ha='center', va='bottom', fontsize=10)

        # Optimize layout
        fig.tight_layout()
        st.pyplot(fig)


        with st.expander("ℹ️ How is the IB Diploma Pass Rate Calculated?"):
            st.markdown("""
            **Criteria Used:**

            - ✅ **Complete 6 subjects** and receive grades for all.
            - ✅ **Minimum of 24 total points** (sum of best 6 subject grades + EE/TOK bonus).
            - ❌ **No grade of 1** in any subject.
            - ⚠️ **No more than one grade of 2**.
            - ⚠️ **No more than three grades of 3 or below**.
            - ➕ **Bonus points** (0 to 3) from the Extended Essay and Theory of Knowledge (EE/TOK) matrix are added.

            
            """)

    elif selected_trend == "📚 IB Group-Wise Subject Trends":
        st.markdown("### 📚 IB Group-Wise Subject Trends")
        with st.expander("ℹ️ IB Subject Group Mapping Reference"):
            st.markdown("""
            ### **Group 1: Studies in Language and Literature**
            - ENGLISH A: Language and Literature  
            - ENGLISH A: Literature  
            - ENGLISH A LIT (P)  
            - SPANISH A: Language and Literature  
            - FRENCH A: Language and Literature  
            - FRENCH A: Literature  
            - CHINESE A: Language and Literature  
            - CHINESE A: Literature  
            - GERMAN A: Language and Literature  
            - GERMAN A: Literature  
            - ITALIAN A: Literature  
            - DUTCH A: Literature  
            - NORWEGIAN A: Literature  
            - PORTUGUESE A: Literature  
            - VIETNAMESE A: Literature  

            ### **Group 2: Language Acquisition**
            - FRENCH B  
            - SPANISH B  
            - GERMAN B  
            - ITALIAN B  
            - ENGLISH B  
            - CHINESE B  
            - ARABIC B  
            - FRENCH AB INITIO.  
            - SPANISH AB INITIO.  
            - ARABIC AB INITIO.  
            - MANDARIN AB INITIO.  

            ### **Group 3: Individuals and Societies**
            - ECONOMICS  
            - ECONOMICS (P)  
            - BUSINESS AND MANAGEMENT  
            - BUS MAN  
            - HISTORY  
            - GEOGRAPHY  
            - PSYCHOLOGY  
            - PSYCHOLOGY (P)  
            - PHILOSOPHY  
            - GLOB. POL  
            - INFORMATION TECHNOLOGY IN GLOBAL SOCIETY  

            ### **Group 4: Sciences**
            - BIOLOGY  
            - CHEMISTRY  
            - PHYSICS  
            - ENVIRONMENTAL SYSTEMS AND SOCIETIES  
            

            ### **Group 5: Mathematics**
            - MATH ANALYSIS  
            - MATH APPS  
            - MATH.STUDIES  
            - MATHEMATICS  

            ### **Group 6: The Arts**
            - VISUAL ARTS  
            - THEATRE  
            - MUSIC  
            - FILM
            - DESIGN TECHNOLOGY  
            - COMPUTER SCIENCE    
            """)

        # Filter out rows with valid group
        grouped_df = df1[df1["Group"].notna()]
        grouped_df["Group"] = grouped_df["Group"].astype(int)

        st.markdown("#### 1. Average Grade by IB Group Over Time")
        group_avg = grouped_df.groupby(["Year", "Group"])["Subject grade"].mean().unstack()
        fig, ax = plt.subplots(figsize=(9, 4))
        group_avg.plot(ax=ax, marker='o')
        ax.set_title("Average Subject Grade by IB Group")
        ax.set_ylabel("Average Grade")
        ax.set_xlabel("Year")
        ax.set_ylim(0, 7)
        ax.grid(True)
        st.pyplot(fig)

        st.markdown("#### 2. Group Comparison for a Selected Year")
        year_selected = st.slider("Select Year", int(df1["Year"].min()), int(df1["Year"].max()), 2024)
        year_data = grouped_df[grouped_df["Year"] == year_selected]
        group_means = year_data.groupby("Group")["Subject grade"].mean()
        fig, ax = plt.subplots(figsize=(7, 3))
        group_means.plot(kind="bar", color="cornflowerblue", ax=ax)
        ax.set_title(f"Average Subject Grade by Group in {year_selected}")
        ax.set_ylabel("Average Grade")
        ax.set_xlabel("IB Group")
        ax.set_ylim(0, 7)
        st.pyplot(fig)

        st.markdown("#### 3. Student Distribution Across IB Groups")
        dist = grouped_df.groupby(["Year", "Group"])["Name"].nunique().unstack().fillna(0)
        fig, ax = plt.subplots(figsize=(9, 4))
        dist.plot(kind="line", marker="o", ax=ax)
        ax.set_title("Student Distribution by IB Group Over Time")
        ax.set_ylabel("Number of Unique Students")
        ax.set_xlabel("Year")
        st.pyplot(fig)

        st.markdown("#### 4. HL vs SL Comparison by Group")
        hl_sl = grouped_df.groupby(["Group", "Level"])["Subject grade"].mean().unstack()
        fig, ax = plt.subplots(figsize=(7, 4))
        hl_sl.plot(kind="bar", ax=ax)
        ax.set_title("Average Grade by Group and Level")
        ax.set_ylabel("Average Grade")
        ax.set_xlabel("IB Group")
        ax.set_ylim(0, 7)
        ax.grid(True)
        st.pyplot(fig)


    elif selected_trend == "📤 Compare School vs World Averages (Upload PDFs)":
        st.markdown("### 📤 Compare School vs World Averages (Upload PDFs)")

        import re
        import numpy as np

        def normalize(name):
            name = name.upper().strip()
            name = re.sub(r"\(P\d{2}\)", "", name)
            name = name.replace("&", "AND")
            name = re.sub(r"[^A-Z0-9 ]", "", name)
            name = re.sub(r"\s+", " ", name)
            return name

        csv_df = pd.read_csv("student_points_df.csv")
        csv_df["Canonical Subject"] = (
            csv_df["Subject"].astype(str).str.strip() + " " +
            csv_df["Level"].astype(str).str.strip() + " " +
            csv_df["Language"].astype(str).str.strip()
        )
        canonical_lookup = {
            normalize(s): s for s in csv_df["Canonical Subject"].unique()
        }

        subject_to_group = {}
        for subj, group in group_mapping.items():
            subject_to_group[normalize(subj)] = group

        group_map = {
            1: "Studies in Language and Literature",
            2: "Language Acquisition",
            3: "Individuals and Societies",
            4: "Sciences",
            5: "Mathematics",
            6: "The Arts"
        }

        if "uploaded_subject_data" not in st.session_state:
            st.session_state.uploaded_subject_data = {}

        year_input = st.text_input("Year of the Uploaded PDF", value="2024")
        uploaded_pdf = st.file_uploader("Upload IB Subject Results PDF", type="pdf")

        def extract_subject_table(pdf_file):
            subject_data = []
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        lines = text.split("\n")
                        for line in lines:
                            # Target lines with average school and world grades near the end
                            parts = re.split(r'\s{2,}', line.strip())
                            if len(parts) >= 4:
                                subject_part = parts[0]
                                school_avg_match = re.search(r'(\d\.\d{2})', line)
                                all_floats = re.findall(r'(\d\.\d{2})', line)

                                if school_avg_match and len(all_floats) >= 2:
                                    raw_subject = subject_part
                                    avg_school = float(all_floats[-2])
                                    avg_world = float(all_floats[-1])

                                    norm_subject = normalize(raw_subject)
                                    closest = get_close_matches(norm_subject, list(canonical_lookup.keys()), n=1, cutoff=0.85)
                                    if closest:
                                        matched = canonical_lookup[closest[0]]
                                        base_subject = normalize(" ".join(matched.split()[:-2]))  # drop HL/SL + Language
                                        subject_data.append({
                                            "Subject": matched,
                                            "Base": base_subject,
                                            "Avg School": avg_school,
                                            "Avg World": avg_world
                                        })

            return pd.DataFrame(subject_data)


        if uploaded_pdf and year_input:
            import pdfplumber
            import re
            import pandas as pd
            from difflib import get_close_matches

            def normalize(name):
                name = name.upper().strip()
                name = re.sub(r"\(P\d{2}\)", "", name)
                name = name.replace("&", "AND")
                name = re.sub(r"[^A-Z0-9 ]", "", name)
                name = re.sub(r"\s+", " ", name)
                return name

            def extract_subject_data_regex(pdf_file):
                subject_data = []

                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            lines = text.split("\n")
                            for line in lines:
                                # Looks for a line like: SUBJECT NAME   20  ... 5.22  5.01
                                match = re.search(r'(.+?)\s+(\d+)\s+.*?(\d\.\d{2})\s+(\d\.\d{2})', line)
                                if match:
                                    subject_name = match.group(1).strip()
                                    avg_school = float(match.group(3))
                                    avg_world = float(match.group(4))
                                    subject_data.append({
                                        "Raw": subject_name,
                                        "Base": normalize(subject_name),
                                        "Avg School": avg_school,
                                        "Avg World": avg_world
                                    })

                return pd.DataFrame(subject_data)

            df_parsed = extract_subject_data_regex(uploaded_pdf)
            if not df_parsed.empty:
                st.session_state.uploaded_subject_data[int(year_input)] = df_parsed
                st.success(f"✅ Data for {year_input} uploaded successfully.")
            else:
                st.warning("⚠️ No valid data extracted from PDF.")

        if st.session_state.uploaded_subject_data:
            selected_year = st.selectbox("Select Year", sorted(st.session_state.uploaded_subject_data.keys()), index=0)
            group_options = {
                1: "1 - Studies in Language and Literature",
                2: "2 - Language Acquisition",
                3: "3 - Individuals and Societies",
                4: "4 - Sciences",
                5: "5 - Mathematics",
                6: "6 - The Arts"
            }
            selected_group = st.selectbox("Select IB Group", options=group_options.keys(), format_func=lambda k: group_options[k])

            df_uploaded = st.session_state.uploaded_subject_data[selected_year]
            group_subjects = [subj for subj, grp in group_mapping.items() if grp == selected_group]
            group_subjects_norm = [normalize(s) for s in group_subjects]
            filtered = df_uploaded[df_uploaded["Base"].isin(group_subjects_norm)]

            if not filtered.empty:
                st.markdown(f"#### 📊 School vs World Averages for {group_options[selected_group]} ({selected_year})")
                fig, ax = plt.subplots(figsize=(11, 4))

                bar_width = 0.4
                index = np.arange(len(filtered))

                ax.bar(index, filtered["Avg School"], width=bar_width, label="School", color="cornflowerblue")
                ax.bar(index + bar_width, filtered["Avg World"], width=bar_width, label="World", color="orange")

                ax.set_xticks(index + bar_width / 2)
                ax.set_xticklabels(filtered["Subject"], rotation=90)
                ax.set_ylabel("Average Grade")
                ax.set_ylim(0, 7)
                ax.set_title("School vs World Averages by Subject")
                ax.legend()

                # Add value labels
                for i, row in filtered.iterrows():
                    ax.text(i, row["Avg School"] + 0.1, f"{row['Avg School']:.2f}", ha='center', fontsize=8)
                    ax.text(i + bar_width, row["Avg World"] + 0.1, f"{row['Avg World']:.2f}", ha='center', fontsize=8)

                st.pyplot(fig)

                # Table
                st.markdown("### 📊 Comparison Table")
                filtered["Grade Gap"] = filtered["Avg School"] - filtered["Avg World"]
                st.dataframe(filtered[["Subject", "Avg School", "Avg World", "Grade Gap"]].style.format({
                    "Avg School": "{:.2f}", "Avg World": "{:.2f}", "Grade Gap": "{:+.2f}"
                }).highlight_max("Avg School", color="lightgreen")
                .highlight_min("Avg World", color="salmon"), use_container_width=True)
            else:
                st.warning("❌ No subjects from this group found in uploaded PDF.")

