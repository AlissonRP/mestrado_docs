from use_cases.validate_clientes import SingleClientValidatorUseCase


class DataFrameValidatorUseCase:
    def __init__(self):
        self.row_validator = SingleClientValidatorUseCase()

    def validate_dataframe(self, df):
        results = []
        for _, row in df.iterrows():
            result = self.row_validator.validate(row)
            results.append(result)
        return results