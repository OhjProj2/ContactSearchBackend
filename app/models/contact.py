from pydantic import BaseModel, create_model
from app.config import Settings
from dataclasses import dataclass

settings = Settings()


@dataclass
class ContactField:
    name: str
    var_name: str


# fields that are used in contact details dropdown list in frontend
default_fields = [
    ContactField("Occupation", "occupation"),
    ContactField("Role", "role"),
    ContactField("Organization", "organization"),
    ContactField("First name", "first_name"),
    ContactField("Last name", "last_name"),
    ContactField("Email", "email"),
    ContactField("Telephone", "telephone"),
    ContactField("Street address", "street_address"),
    ContactField("Postal code", "postal_code"),
    ContactField("City", "city"),
    ContactField("Country", "country"),
    ContactField("Web page", "web_page"),
    ContactField("Social media (default list)", "social_media"),
    ContactField("Business ID", "business_id"),
    ContactField("Company name", "company_name"),
    ContactField("Non-governmental organization name", "ngo_name"),
    ContactField("Political party", "political_party"),
]

# a dictionary mapping used in converting explanatory names to variable/data field names
field_map = {field.name: field.var_name for field in default_fields}


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
    """Takes list of contact details and changes it to a dictionary
    format the Pydantics create_model method can use to create
    a ContactList class.

    Args:
        contact_details: list of contact details to be searched from fetched data

    Returns:
        contact_dict: dictionary in correct format for create_model method
    """
    contact_dict = {}
    # map contact details that match the default list to
    # default variable names
    contact_details = [
        field_map.get(contact_detail, contact_detail)
        for contact_detail in contact_details
    ]
    # tidy contact details to all lowercase and replace
    # spaces with underscores
    contact_details = [
        contact_detail.lower().replace(" ", "_") for contact_detail in contact_details
    ]
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
        db_uri: MongoDB Universal Resource Identifier
        db_name: MongoDB database name (one db per one project)
        db_collection: MongoDB collection name (one collection per one kind of contact details)
    """

    contact_details: list[str]
    occupations: list[str]
    url: str
    temp: float | None = settings.OLLAMA_TEMPERATURE
    top_p: float | None = settings.OLLAMA_TOP_P
    num_predict: int | None = settings.OLLAMA_NUM_PREDICT
    num_ctx: int | None = settings.OLLAMA_NUM_CTX
    model: str | None = settings.OLLAMA_MODEL
    db_uri: str | None = settings.MONGODB_URI
    db_name: str | None = settings.MONGODB_NAME
    db_collection: str | None = settings.MONGODB_COLLECTION


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
