from use_cases.validate_dataframe import DataFrameValidatorUseCase
import pandas as pd


df = pd.DataFrame([
        {"cliente_id": 1, "ticket": "ok@email.com", "address": "user@email.com"},
        {"cliente_id": 2, "ticket": "", "address": "semarroba.com"},
    ])


def test_validate_multiple_clients():
    df = pd.DataFrame([
        {"cliente_id": 1, "ticket": "ok@email.com", "address": "user@email.com"},
        {"cliente_id": 2, "ticket": "", "address": "semarroba.com"},
    ])

    validator = DataFrameValidatorUseCase()
    results = validator.validate_dataframe(df)

    results[0]['ticket_is_valid'] is True
    results[1]['ticket_is_valid'] is False