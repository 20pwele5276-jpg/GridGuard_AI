import hashlib
from io import BytesIO

import pandas as pd
import streamlit as st

from services.anomaly_detector import detect_anomalies
from services.report_generator import generate_pdf_report
from services.manual_search import (
    extract_manual_chunks,
    build_manual_index,
    search_manual,
)
from services.llm_service import answer_manual_question


st.set_page_config(
    page_title="GridGuard AI",
    page_icon="⚡",
    layout="wide",
)

REQUIRED_COLUMNS = [
    "timestamp",
    "equipment_id",
    "voltage_v",
    "current_a",
    "frequency_hz",
    "power_factor",
    "temperature_c",
]


def create_demo_data():
    rows = [
        ["2026-10-09 08:00", "Transformer-01", 231, 14.1, 50.0, 0.96, 42],
        ["2026-10-09 08:05", "Transformer-01", 233, 14.5, 50.0, 0.95, 43],
        ["2026-10-09 08:10", "Transformer-01", 230, 14.0, 50.1, 0.96, 42],
        ["2026-10-09 08:15", "Transformer-01", 236, 15.2, 50.0, 0.94, 46],
        ["2026-10-09 08:20", "Transformer-01", 240, 16.1, 50.0, 0.92, 50],
        ["2026-10-09 08:25", "Transformer-01", 252, 18.4, 50.2, 0.87, 63],
        ["2026-10-09 08:30", "Transformer-01", 258, 19.2, 50.1, 0.84, 78],
        ["2026-10-09 08:35", "Transformer-01", 254, 18.9, 49.9, 0.85, 84],
        ["2026-10-09 08:40", "Transformer-01", 239, 16.0, 50.0, 0.92, 61],
        ["2026-10-09 08:45", "Transformer-01", 234, 14.8, 50.0, 0.95, 48],
        ["2026-10-09 08:50", "Transformer-01", 232, 14.3, 50.0, 0.96, 44],
        ["2026-10-09 08:55", "Transformer-01", 231, 14.1, 50.0, 0.96, 43],
    ]

    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def create_template():
    return pd.DataFrame(
        [[
            "2026-10-09 08:00",
            "Transformer-01",
            231,
            14.1,
            50.0,
            0.96,
            42,
        ]],
        columns=REQUIRED_COLUMNS,
    )


def load_uploaded_file(uploaded_file):
    file_data = BytesIO(uploaded_file.getvalue())

    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(file_data)

    return pd.read_excel(file_data)


def run_analysis(dataframe, source_name, state_key):
    st.session_state.pop(state_key, None)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        st.error(
            "Required columns are missing: "
            + ", ".join(missing_columns)
        )
        return

    if len(dataframe) < 5:
        st.error("Please provide at least 5 readings for analysis.")
        return

    try:
        with st.spinner("Analyzing electrical readings..."):
            result = detect_anomalies(dataframe)

        st.session_state[state_key] = {
            "data": result,
            "source": source_name,
        }

    except Exception as error:
        st.error(f"Analysis failed: {error}")


def show_analysis(analysis):
    result = analysis["data"]
    source_name = analysis["source"]

    flagged = result[
        result["anomaly_status"] == "Potential anomaly"
    ]

    total = len(result)
    flagged_count = len(flagged)

    st.divider()
    st.subheader("Analysis Results")

    first, second, third = st.columns(3)

    with first:
        st.metric("Readings Analyzed", total)

    with second:
        st.metric("Readings Flagged", flagged_count)

    with third:
        st.metric(
            "Share Flagged",
            f"{flagged_count / total * 100:.1f}%",
        )

    if flagged_count:
        st.warning(
            f"{flagged_count} readings differ from patterns "
            "in this dataset. Review them before making decisions."
        )
    else:
        st.success("No readings were flagged by this model.")

    st.subheader("Detection Results")
    st.dataframe(result, width="stretch")

    if flagged_count:
        st.subheader("Readings to Review")
        st.dataframe(flagged, width="stretch")

    chart_data = result.copy()
    chart_data["timestamp"] = pd.to_datetime(
        chart_data["timestamp"],
        errors="coerce",
    )

    chart_data = chart_data.dropna(
        subset=["timestamp"]
    ).sort_values("timestamp")

    if not chart_data.empty:
        st.subheader("Voltage Trend")
        st.line_chart(
            chart_data.set_index("timestamp")[["voltage_v"]]
        )

        st.subheader("Temperature Trend")
        st.line_chart(
            chart_data.set_index("timestamp")[["temperature_c"]]
        )
    else:
        st.info(
            "Trend charts are unavailable because the timestamps "
            "could not be interpreted as dates."
        )

    st.subheader("Download Reports")

    csv_data = result.to_csv(index=False).encode("utf-8")
    pdf_data = generate_pdf_report(
        result,
        source_name=source_name,
    )

    left, right = st.columns(2)

    with left:
        st.download_button(
            "Download CSV Analysis",
            data=csv_data,
            file_name="gridguard_analysis.csv",
            mime="text/csv",
            width="stretch",
            key=f"{source_name}_csv",
        )

    with right:
        st.download_button(
            "Download PDF Report",
            data=pdf_data,
            file_name="gridguard_maintenance_report.pdf",
            mime="application/pdf",
            width="stretch",
            key=f"{source_name}_pdf",
        )

    st.caption(
        "The anomaly score is a relative model score, not a "
        "probability of equipment failure."
    )


def show_manual_assistant():
    st.divider()
    st.header("Equipment Manual Assistant")

    st.write(
        "Upload an equipment manual and ask a question about its "
        "maintenance instructions or specifications."
    )

    manual_pdf = st.file_uploader(
        "Upload an equipment manual (PDF)",
        type=["pdf"],
        key="manual_pdf_upload",
    )

    question = st.text_area(
        "What would you like to know?",
        placeholder=(
            "Example: What routine inspection checks does the "
            "manual recommend?"
        ),
        key="manual_question_input",
    )

    if st.button(
        "Search Manual and Explain",
        type="primary",
        width="stretch",
        key="search_manual_button",
    ):
        if manual_pdf is None:
            st.warning("Please upload a PDF manual first.")

        elif not question.strip():
            st.warning("Please enter a question.")

        else:
            file_bytes = manual_pdf.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            st.session_state.pop("manual_search_result", None)

            try:
                with st.spinner(
                    "Reading the manual and finding relevant passages..."
                ):
                    chunks = extract_manual_chunks(
                        file_bytes,
                        manual_pdf.name,
                    )

                    index = build_manual_index(chunks)

                    evidence = search_manual(
                        chunks,
                        index,
                        question.strip(),
                        top_k=4,
                    )

                with st.spinner(
                    "Preparing an explanation from the manual..."
                ):
                    explanation = answer_manual_question(
                        question.strip(),
                        evidence,
                    )

                st.session_state["manual_search_result"] = {
                    "source": manual_pdf.name,
                    "file_hash": file_hash,
                    "question": question.strip(),
                    "evidence": evidence,
                    "explanation": explanation,
                }

            except Exception as error:
                st.error(f"Manual search failed: {error}")

    saved = st.session_state.get("manual_search_result")

    if manual_pdf is None or not question.strip() or saved is None:
        return

    current_hash = hashlib.sha256(
        manual_pdf.getvalue()
    ).hexdigest()

    if (
        saved["file_hash"] != current_hash
        or saved["question"] != question.strip()
    ):
        return

    st.subheader("Explanation")
    st.markdown(saved["explanation"])

    st.subheader("Retrieved Manual Passages")

    for number, item in enumerate(saved["evidence"], start=1):
        similarity = item.get("similarity", 0.0)

        with st.expander(
            f"Passage {number} | Page {item['page']} "
            f"| Similarity {similarity:.0%}"
        ):
            st.caption(f"Source: {item['source']}")
            st.write(item["text"])

    st.caption(
        "Check the original manual before acting on the explanation. "
        "This feature does not authorize electrical work or replace "
        "a qualified professional's assessment."
    )


def show_header():
    st.title("GridGuard AI")
    st.caption(
        "Electrical equipment monitoring and maintenance support"
    )

    dashboard, help_page, sample_page = st.columns(3)

    with dashboard:
        if st.button("Dashboard", width="stretch"):
            st.session_state.page = "dashboard"
            st.rerun()

    with help_page:
        if st.button("How to Use", width="stretch"):
            st.session_state.page = "help"
            st.rerun()

    with sample_page:
        if st.button("Try Sample Analysis", width="stretch"):
            st.session_state.page = "sample"
            st.rerun()

    st.divider()


def show_dashboard():
    st.header("Analyze Your Equipment Data")

    st.write(
        "Upload electrical readings to check for unusual patterns "
        "and download a maintenance review report."
    )

    uploaded_file = st.file_uploader(
        "Upload your CSV or Excel file",
        type=["csv", "xlsx"],
        help=(
            "Use the column names shown on the How to Use page. "
            "At least five measurement rows are required."
        ),
        key="user_upload",
    )

    if uploaded_file is None:
        st.info(
            "You can analyze your own data here, or open "
            "Try Sample Analysis to explore GridGuard first."
        )

        st.download_button(
            "Download CSV Template",
            data=create_template().to_csv(index=False),
            file_name="gridguard_template.csv",
            mime="text/csv",
            key="dashboard_template",
        )

        st.session_state.pop("user_analysis", None)

    else:
        try:
            dataframe = load_uploaded_file(uploaded_file)

            st.subheader("File Preview")
            st.caption(f"Selected file: {uploaded_file.name}")
            st.dataframe(
                dataframe.head(10),
                width="stretch",
            )

            if st.button(
                "Analyze My Data",
                type="primary",
                width="stretch",
            ):
                run_analysis(
                    dataframe,
                    uploaded_file.name,
                    "user_analysis",
                )

            saved_analysis = st.session_state.get("user_analysis")

            if (
                saved_analysis
                and saved_analysis["source"] == uploaded_file.name
            ):
                show_analysis(saved_analysis)

        except Exception as error:
            st.error(f"Could not read this file: {error}")

    show_manual_assistant()


def show_help():
    st.header("How to Use GridGuard")

    st.write(
        "Follow this guide if you are using GridGuard for the first time."
    )

    with st.container(border=True):
        st.subheader("1. Prepare your readings file")

        st.write(
            "Export your electrical measurements to CSV or Excel. "
            "Use one row for each measurement."
        )

        st.code("\n".join(REQUIRED_COLUMNS))

        st.download_button(
            "Download CSV Template",
            data=create_template().to_csv(index=False),
            file_name="gridguard_template.csv",
            mime="text/csv",
            key="help_template",
        )

    with st.container(border=True):
        st.subheader("2. Understand the columns")

        st.markdown(
            """
            - **timestamp:** date and time of the measurement.
            - **equipment_id:** equipment name or identifier.
            - **voltage_v:** voltage in volts.
            - **current_a:** current in amperes.
            - **frequency_hz:** frequency in hertz.
            - **power_factor:** power factor, normally from 0 to 1.
            - **temperature_c:** temperature in degrees Celsius.
            """
        )

        st.write(
            "Use the exact column names shown above and provide "
            "at least five readings."
        )

    with st.container(border=True):
        st.subheader("3. Analyze the readings")

        st.write(
            "Return to Dashboard, upload your file, preview the data, "
            "and click Analyze My Data. Review flagged readings and "
            "the voltage and temperature charts."
        )

    with st.container(border=True):
        st.subheader("4. Ask a question about an equipment manual")

        st.write(
            "On the Dashboard, upload a text-based PDF maintenance "
            "manual, enter a question, and click Search Manual and "
            "Explain. Check the cited pages in the original document."
        )

    with st.container(border=True):
        st.subheader("5. Save your results")

        st.write(
            "Download the analysis as CSV or a maintenance review "
            "report as PDF."
        )

    st.warning(
        "GridGuard provides decision support. It does not establish "
        "that equipment is faulty or replace inspection by a qualified "
        "electrical professional."
    )


def show_sample_page():
    st.header("Try Sample Analysis")

    st.write(
        "Explore GridGuard with example measurements. These readings "
        "are for demonstration only and do not represent a real site."
    )

    if st.button(
        "Run Sample Analysis",
        type="primary",
        width="stretch",
    ):
        run_analysis(
            create_demo_data(),
            "GridGuard sample data",
            "sample_analysis",
        )

    sample_analysis = st.session_state.get("sample_analysis")

    if sample_analysis:
        show_analysis(sample_analysis)
    else:
        st.info(
            "Click Run Sample Analysis to generate the example results."
        )


if "page" not in st.session_state:
    st.session_state.page = "dashboard"


show_header()

if st.session_state.page == "dashboard":
    show_dashboard()

elif st.session_state.page == "help":
    show_help()

elif st.session_state.page == "sample":
    show_sample_page()

st.divider()

st.caption(
    "GridGuard AI | Electrical monitoring and maintenance support"
)