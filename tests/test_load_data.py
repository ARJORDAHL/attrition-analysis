import pandas as pd
import pytest
from src.load_data import clean_employee_data


def make_base_df(**overrides):
    data = {
        "employee_id": [1, 2],
        "department": ["Sales", "HR"],
        "age": [30, 40],
        "monthly_income": [5000.0, 6000.0],
        "job_satisfaction": [3, 4],
        "overtime": ["Yes", "No"],
        "travel_frequency": ["Frequent", "Rarely"],
        "years_at_company": [3, 7],
        "attrition": ["Yes", "No"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def test_clean_raises_on_missing_column():
    df = make_base_df().drop(columns=["department"])
    with pytest.raises(ValueError, match="Missing required columns"):
        clean_employee_data(df)


def test_clean_fills_missing_department_with_unknown():
    df = make_base_df(department=[None, "HR"])
    result = clean_employee_data(df)
    assert result["department"].iloc[0] == "Unknown"


def test_clean_fills_missing_overtime_with_no():
    df = make_base_df(overtime=[None, "No"])
    result = clean_employee_data(df)
    assert result["overtime"].iloc[0] == "No"


def test_clean_fills_missing_travel_frequency_with_rarely():
    df = make_base_df(travel_frequency=[None, "Frequent"])
    result = clean_employee_data(df)
    assert result["travel_frequency"].iloc[0] == "Rarely"


def test_clean_fills_missing_job_satisfaction_with_3():
    df = make_base_df(job_satisfaction=[None, 4])
    result = clean_employee_data(df)
    assert result["job_satisfaction"].iloc[0] == 3


def test_clean_fills_missing_monthly_income_with_median():
    df = make_base_df(monthly_income=[None, 6000.0])
    result = clean_employee_data(df)
    assert result["monthly_income"].iloc[0] == 6000.0


def test_clean_normalizes_attrition_to_title_case():
    df = make_base_df(attrition=["YES", "no"])
    result = clean_employee_data(df)
    assert list(result["attrition"]) == ["Yes", "No"]


def test_clean_strips_whitespace_from_department():
    df = make_base_df(department=["  Sales  ", "HR"])
    result = clean_employee_data(df)
    assert result["department"].iloc[0] == "Sales"


def test_clean_does_not_modify_original_dataframe():
    df = make_base_df(department=[None, "HR"])
    clean_employee_data(df)
    assert pd.isna(df["department"].iloc[0])
