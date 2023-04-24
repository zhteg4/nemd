import dash


def callback(*arg, **kwargs):
    # print(arg, kwargs)
    return dash.callback(*arg, **kwargs)


class Upload(dash.dcc.Upload):
    """
    Upload component with customized style.
    """

    STYLE_KEY = 'style'
    BORDERWIDTH = 'borderWidth'
    BORDERSTYLE = 'borderStyle'
    DASHED = 'dashed'
    BORDERRADIUS = 'borderRadius'
    TEXTALIGN = 'textAlign'
    CENTER = 'center'

    STYLE = {
        BORDERWIDTH: '1px',
        BORDERSTYLE: DASHED,
        BORDERRADIUS: '5px',
        TEXTALIGN: CENTER,
        'padding-left': 10,
        'padding-right': 10
    }

    def __init__(self, *args, **kwargs):
        style = {**self.STYLE, **kwargs.pop(self.STYLE_KEY, {})}
        kwargs[self.STYLE_KEY] = style
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
