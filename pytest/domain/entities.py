class Ticket:
    def __init__(self, value: str):
        self.value = value.strip() if value else ""

    def is_too_long(self):
        return len(self.value) > 250
    
    def is_empty(self):
        return self.value == ''
    
    def is_valid(self):

        return not self.is_empty() and not self.is_too_long()
    


class Address:
    def __init__(self, value: str):
        self.value = value.strip() if value else ""

    def contains_at_symbol(self):
        return "@" in self.value

    def is_valid(self):
        return self.contains_at_symbol()
