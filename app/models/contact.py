from pydantic import BaseModel, create_model
from config import Settings

settings = Settings()


def _contact_type(contact_detail: str) -> tuple:
    if contact_detail == "social_media":
        return (SocialMedia, ...)
    else:
        return (str, ...)


def _create_contact_dict(contact_details: list[str]) -> dict:
    contact_dict = {}
    for c in contact_details:
        contact_dict[c] = _contact_type(c)
    return contact_dict


class SocialMedia(BaseModel):
    linkedin: str
    twitter_x: str
    facebook: str
    telegram: str
    signal: str
    instagram: str


class SeekParameters(BaseModel):
    contact_details: list[str]
    occupations: list[str]
    url: str
    temp: float | None = settings.OLLAMA_TEMPERATURE
    top_p: float | None = settings.OLLAMA_TOP_P
    num_predict: int | None = settings.OLLAMA_NUM_PREDICT
    num_ctx: int | None = settings.OLLAMA_NUM_CTX
    model: str | None = settings.OLLAMA_MODEL


def build_contact_list_model(contact_details: list[str]) -> type[BaseModel]:
    contact_dict = _create_contact_dict(contact_details)
    ContactInfo = create_model("ContactInfo", **contact_dict)

    class ContactList(BaseModel):
        contacts: list[ContactInfo]

    return ContactList
