from use_cases.validate_clientes import SingleClientValidatorUseCase
import pandas as pd








def test_valida_ticket_e_address():
    # cria DataFrame de exemplo
    data = pd.DataFrame([
        {"cliente_id": 1, "ticket": "teste@email.com", "address": "teste@email.com"},
        #{"cliente_id": 2, "ticket": "", "address": "semarroba.com"},
        #{"cliente_id": 3, "ticket": "a" * 251 + "@gmail.com", "address": "ok@ok.com"},
    ])
    df = data.iloc[0]
    use_case = SingleClientValidatorUseCase()
    result = use_case.validate(df)

    assert len(result) != 3 # erro vai ser 1


    # cliente 1: tudo válido
    assert result["ticket_is_valid"] is True
    assert result["address_is_valid"] is True
