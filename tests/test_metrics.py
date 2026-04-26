import pandas as pd
import pytest
from src.metrics import (
    attrition_rate,
    attrition_by_department,
    attrition_by_overtime,
    average_income_by_attrition,
    satisfaction_summary,
)


def make_df():
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 6],
        "department": ["Sales", "Sales", "HR", "HR", "IT", "IT"],
        "overtime": ["Yes", "Yes", "No", "No", "Yes", "No"],
        "monthly_income": [4000, 6000, 5000, 7000, 4500, 8000],
        "job_satisfaction": [1, 2, 3, 4, 2, 3],
        "attrition": ["Yes", "Yes", "No", "No", "Yes", "No"],
    })


# --- attrition_rate ---

def test_attrition_rate_returns_expected_percent():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "attrition": ["Yes", "No", "No", "Yes"],
    })
    assert attrition_rate(df) == 50.0


def test_attrition_rate_all_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["Yes", "Yes"],
    })
    assert attrition_rate(df) == 100.0


def test_attrition_rate_no_leavers():
    df = pd.DataFrame({
        "employee_id": [1, 2],
        "attrition": ["No", "No"],
    })
    assert attrition_rate(df) == 0.0


# --- attrition_by_department ---

def test_attrition_by_department_returns_expected_columns():
    result = attrition_by_department(make_df())
    assert list(result.columns) == ["department", "employees", "leavers", "attrition_rate"]


def test_attrition_by_department_computes_correct_rates():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "department": ["Sales", "Sales", "HR", "HR"],
        "attrition": ["Yes", "Yes", "No", "No"],
    })
    result = attrition_by_department(df)
    sales = result[result["department"] == "Sales"].iloc[0]
    hr = result[result["department"] == "HR"].iloc[0]
    assert sales["attrition_rate"] == 100.0
    assert hr["attrition_rate"] == 0.0


def test_attrition_by_department_sorted_descending():
    result = attrition_by_department(make_df())
    rates = list(result["attrition_rate"])
    assert rates == sorted(rates, reverse=True)


# --- attrition_by_overtime ---

def test_attrition_by_overtime_returns_expected_columns():
    result = attrition_by_overtime(make_df())
    assert list(result.columns) == ["overtime", "employees", "leavers", "attrition_rate"]


def test_attrition_by_overtime_computes_correct_rates():
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "overtime": ["Yes", "Yes", "No", "No"],
        "attrition": ["Yes", "Yes", "No", "No"],
    })
    result = attrition_by_overtime(df)
    yes_row = result[result["overtime"] == "Yes"].iloc[0]
    no_row = result[result["overtime"] == "No"].iloc[0]
    assert yes_row["attrition_rate"] == 100.0
    assert no_row["attrition_rate"] == 0.0


# --- average_income_by_attrition ---

def test_average_income_by_attrition_returns_expected_columns():
    result = average_income_by_attrition(make_df())
    assert list(result.columns) == ["attrition", "avg_monthly_income"]


def test_average_income_by_attrition_computes_correct_averages():
    df = pd.DataFrame({
        "attrition": ["Yes", "Yes", "No", "No"],
        "monthly_income": [3000, 5000, 6000, 8000],
    })
    result = average_income_by_attrition(df)
    yes_row = result[result["attrition"] == "Yes"].iloc[0]
    no_row = result[result["attrition"] == "No"].iloc[0]
    assert yes_row["avg_monthly_income"] == 4000.0
    assert no_row["avg_monthly_income"] == 7000.0


# --- satisfaction_summary ---

def test_satisfaction_summary_returns_expected_columns():
    result = satisfaction_summary(make_df())
    assert list(result.columns) == ["job_satisfaction", "total_employees", "leavers", "attrition_rate"]


def test_satisfaction_summary_computes_per_group_rate():
    # Validates the fixed denominator: rate = leavers / employees in that group,
    # not leavers / total leavers across all groups.
    df = pd.DataFrame({
        "employee_id": [1, 2, 3, 4],
        "job_satisfaction": [1, 1, 4, 4],
        "attrition": ["Yes", "Yes", "No", "No"],
    })
    result = satisfaction_summary(df)
    sat1 = result[result["job_satisfaction"] == 1].iloc[0]
    sat4 = result[result["job_satisfaction"] == 4].iloc[0]
    assert sat1["attrition_rate"] == 100.0
    assert sat4["attrition_rate"] == 0.0


def test_satisfaction_summary_sorted_ascending():
    result = satisfaction_summary(make_df())
    levels = list(result["job_satisfaction"])
    assert levels == sorted(levels)
