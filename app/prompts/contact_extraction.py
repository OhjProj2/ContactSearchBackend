class SystemMessage:
    def __init__(self, occupations: list[str] | None) -> None:
        self.occupations = ", ".join(occupations) if occupations else None
        self.personality = "You are a data extraction assistant specialized in extracting contact information from webpages."
        if self.occupations:
            self.task = f"TASK: Extract contact details of following occupations/roles found in the provided webpage content: ({self.occupations})"
        else:
            self.task = f"TASK: Extract all contact details found in the provided webpage content."
        self.output_format = "OUTPUT FORMAT: Return a valid JSON list of objects."
        self.rules = "RULES:- Construct the email address if one is explicitly provided or implicitly given as a rule of forming it from contact details."

    def create(self):
        sys_message = self.personality + "\n" + self.task + "\n" + self.rules
        return sys_message


class UserPrompt:
    def __init__(self, web_content: str):
        self.prompt = f"Here is the content of a webpage:\n\n{web_content}"

    def create(self):
        return self.prompt


def build_messages(
    system_message: SystemMessage, user_prompt: UserPrompt
) -> list[tuple]:
    return [("system", system_message.create()), ("human", user_prompt.create())]


def main():
    sm = SystemMessage(["Pappi", "Lukkari", "Talonpoika", "Varas"])
    up = UserPrompt("<html>Kotisivu</html>")

    print(sm.create())
    print(up.create())


if __name__ == "__main__":
    main()
