from domain.entities import Ticket, Address


class SingleClientValidatorUseCase:
    def validate(self, row):
        #row = dataframe.iloc[0]
    

        ticket = Ticket(row["ticket"])
        address = Address(row["address"])

        return {
            "cliente_id": row.get("cliente_id"),
            "ticket": row["ticket"],
            "address": row["address"],
            "ticket_is_valid": ticket.is_valid(),
            "ticket_empty": ticket.is_empty(),
            "ticket_too_long": ticket.is_too_long(),
            "address_is_valid": address.is_valid(),
            "address_contains_at": address.contains_at_symbol()
        }


        


