"""Example entity schemas for LLM Anonymizer.

Each BaseModel subclass defines one entity type to detect. The LLM will be
called once per schema per text chunk, using Ollama's structured output.

The entity_type ClassVar is optional — if omitted, it is auto-derived from
the class name (CamelCase → UPPER_SNAKE_CASE).
"""

from typing import ClassVar, List, Union

from pydantic import BaseModel


# class PatientIdentities(BaseModel):
#     """Name of patients. Nothing generic, only what is identifying, including part of name, initials, etc."""
#
#     entity_type: ClassVar[str] = "PERSON"
#     patient_identities: Union[str, List[str]]
#
# class ProfessionalIdentities(BaseModel):
#     """Names of workers, as opposed to patients. Ex: "'Dr. Klaus' said" but NOT if anonymous like "'The doctor" said". Include parts of names, initials, etc."""
#
#     entity_type: ClassVar[str] = "PERSON"
#     professional_identities: Union[str, List[str]]
#
# class NamesOfPeople(BaseModel):
#     """People's identities. Nothing generic, only what is identifying, including part of name, initials, etc."""
#
#     entity_type: ClassVar[str] = "PERSON"
#     names_of_people: Union[str, List[str]]

class EmailAddresses(BaseModel):
    """Email addresses of individuals.
    Extract the full address exactly as written.
    Ex: 'john.doe@example.com', 'j.smith+work@hospital.org'."""

    entity_type: ClassVar[str] = "EMAIL_ADDRESS"
    email_addresses: Union[str, List[str]]


class Addresses(BaseModel):
    """Physical addresses: street, city, postal code, country, or any combination.
    Extract exactly as written, including partial addresses.
    Ex: '221B Baker Street, London', '75008 Paris', 'Rue de la Paix 3, 1000 Bruxelles'.
    NOT vague locations like 'the hospital' or 'downtown'."""

    entity_type: ClassVar[str] = "ADDRESS"
    addresses: Union[str, List[str]]


class People(BaseModel):
    """Names of specific, identifiable people: full names, first names, last names, initials, nicknames. Be creful to include exotic oe complex nobility names.
    Extract only what appears in the text, exactly as written.
    Ex: 'John', 'Dr. Smith', 'M. Dupont', 'J.D.', 'little Timmy'.
    NOT generic roles like 'the doctor', 'the patient', 'a nurse'."""

    entity_type: ClassVar[str] = "PERSON"
    names: Union[str, List[str]]


class PhoneNumbers(BaseModel):
    """Phone numbers in any format, including country code, spaces, dashes, or dots.
    Ex: '+1-800-555-0199', '06 12 34 56 78', '(0)32 123 45 67'.
    NOT extension-only numbers like 'ext. 42'."""

    entity_type: ClassVar[str] = "PHONE_NUMBER"
    phone_numbers: Union[str, List[str]]


class DatesOfBirth(BaseModel):
    """Dates of birth of individuals, in any format.
    Ex: '14/03/1990', 'March 14, 1990', '1990-03-14', 'born in 1990'.
    NOT ages like '35 years old'."""

    entity_type: ClassVar[str] = "DATE_OF_BIRTH"
    dates_of_birth: Union[str, List[str]]

