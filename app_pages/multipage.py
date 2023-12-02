import streamlit as st


# Class to generate multiple Streamlit pages using an object oriented approach
class MultiPage:

    def __init__(self, app_name) -> None:
        """
        Initializes the MultiPage object.

        Parameters:
            app_name (str): The name of the Streamlit app.
        """
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🌱")  # You may add an icon, to personalize your App
        # check links below for additional icons reference
        # https://docs.streamlit.io/en/stable/api.html#streamlit.set_page_config
        # https://twemoji.maxcdn.com/2/test/preview.html

    def add_page(self, title, func) -> None:
        """
        Adds a new page to the app.

        Parameters:
            title (str): The title of the page.
            func (function): The function that generates the content for the
            page.
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """
        Runs the Streamlit app and displays the selected page.
        """
        st.title(self.app_name)
        page = st.sidebar.radio(
            'Menu', self.pages, format_func=lambda page: page['title'])
        page['function']()
