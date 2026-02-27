from pydantic import BaseModel, create_model
from config import Settings

settings = Settings()


def _contact_type(contact_detail: str) -> tuple:
    """Returns the type annotation for a specific contact detail.

    Args:
        contact_detail: Name of the contact detail field.

    Returns:
        Tuple of (type, ...) for the field annotation.
    """
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
    """Default social media service providers extracted from web pages.

    Attributes:
        linkedin: LinkedIn profile
        twitter_x: Twitter/X profile
        facebook: Facebook profile
        telegram: Telegram profile
        signal: Signal profile
        instagram: instagram profile
    """

    linkedin: str
    twitter_x: str
    facebook: str
    telegram: str
    signal: str
    instagram: str


class SeekParameters(BaseModel):
    """Parameters used in calls to seek web page contact details.

    Attributes:
        contact_details: List of contact details to extract
        occupations: List of occupations/roles/titles to search for
        url: URL of the web page to fetch
        temp: Model temperature setting (defaults from settings)
        top_p: Model nucleus sampling settings (defaults from settings)
        num_predict: Max tokens to predict (defaults from settings)
        num_ctx: Context window size (defaults from settings)
        model: Model name (defaults from settings)
    """

    contact_details: list[str]
    occupations: list[str]
    url: str
    temp: float | None = settings.OLLAMA_TEMPERATURE
    top_p: float | None = settings.OLLAMA_TOP_P
    num_predict: int | None = settings.OLLAMA_NUM_PREDICT
    num_ctx: int | None = settings.OLLAMA_NUM_CTX
    model: str | None = settings.OLLAMA_MODEL


def build_contact_list_model(contact_details: list[str]) -> type[BaseModel]:
    """Dynamically builds a ContactList model based on required fields.
    This class is used to get correctly formed structured output from the large language models.

    Args:
        contact_details: Names of contact fields to include in class.

    Returns:
        Pydantic BaseModel class for a list of contacts.
    """
    contact_dict = _create_contact_dict(contact_details)
    ContactInfo: BaseModel = create_model("ContactInfo", **contact_dict)

    class ContactList(BaseModel):
        contacts: list[ContactInfo]

    return ContactList
