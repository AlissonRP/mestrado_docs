from domain.entities import Ticket
import pytest


import logging
logging.basicConfig(level=logging.INFO)

#@pytest.mark.ticket
def test_ticket_vazio():
    logging.info("Rodando teste: test_ticket_vazio")
    ticket = Ticket("")
    assert ticket.is_empty() is True
    assert ticket.is_valid() is False

def test_ticket_espaco_em_branco():
    ticket = Ticket("   ")
    assert ticket.is_empty() is True
    assert ticket.is_valid() is False

def test_ticket_muito_longo():
    ticket = Ticket("a" * 251)
    assert ticket.is_too_long() is True
    assert ticket.is_valid() is False

def test_ticket_valido():
    ticket = Ticket("email@ok.com")
    assert ticket.is_empty() is False
    assert ticket.is_too_long() is False
    assert ticket.is_valid() is True