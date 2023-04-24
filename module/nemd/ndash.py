import dash

TEXTALIGN = 'textAlign'
CENTER = 'center'
COLOR = 'color'
STYLE_KEY = 'style'
PADDING_LEFT = 'padding-left'
WHITE_HEX = '#000000'


class H1(dash.html.H1):
    """
    Header of the website
    """
    BLUE_COLOR_HEX = '#7FDBFF'
    STYLE = {TEXTALIGN: CENTER, COLOR: BLUE_COLOR_HEX}

    def __init__(self, *args, **kwargs):
        style = {**self.STYLE, **kwargs.pop(STYLE_KEY, {})}
        kwargs[STYLE_KEY] = style
        super().__init__(*args, **kwargs)


class LabeledDropdown(dash.html.Div):
    """
    Dropdown list with label.
    """
    STYLE = {PADDING_LEFT: 5, COLOR: WHITE_HEX}

    def __init__(self, *args, label=None, **kwargs):
        self.label = dash.html.Div(children=label)
        style = {**self.STYLE, **kwargs.pop(STYLE_KEY, {})}
        kwargs[STYLE_KEY] = style
        self.dropdown = dash.dcc.Dropdown(*args, **kwargs)
        super().__init__(children=[self.label, self.dropdown])


class Upload(dash.dcc.Upload):
    """
    Upload component with customized style.
    """

    BORDERWIDTH = 'borderWidth'
    BORDERSTYLE = 'borderStyle'
    DASHED = 'dashed'
    BORDERRADIUS = 'borderRadius'

    STYLE = {
        BORDERWIDTH: '1px',
        BORDERSTYLE: DASHED,
        BORDERRADIUS: '5px',
        TEXTALIGN: CENTER,
        'padding-left': 10,
        'padding-right': 10
    }

    def __init__(self, *args, **kwargs):
        style = {**self.STYLE, **kwargs.pop(STYLE_KEY, {})}
        kwargs[STYLE_KEY] = style

        super().__init__(*args, **kwargs)


class LabeledUpload(dash.html.Div):
    """
    Upload component with in-line labels.
    """

    STYLE = {'display': 'inline-block', 'margin-left': 5}

    def __init__(
        self,
        label=None,
        status_id=None,
        button_id=None,
        click_id=None,
    ):
        self.label = dash.html.Div(children=label)
        self.status = dash.html.Div(children='',
                                    id=status_id,
                                    style=self.STYLE)
        button = Upload(
            id=button_id,
            children=dash.html.Div(children='', id=click_id),
        )
        self.button = dash.html.Div(children=button, style=self.STYLE)
        super().__init__(children=[self.label, self.status, self.button])
