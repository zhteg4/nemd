import dash


def callback(*arg, **kwargs):
    # print(arg, kwargs)
    return dash.callback(*arg, **kwargs)


class Upload(dash.dcc.Upload):
    """
    Upload component with customized style.
    """

    STYLE_KEY = 'style'
    STYLE = {
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center'
    }
    BORDERWIDTH = 'borderWidth'
    BORDERSTYLE = 'borderStyle'
    DASHED = 'dashed'
    BORDERRADIUS = 'borderRadius'
    TEXTALIGN = 'textAlign'
    CENTER = 'center'

    def __init__(self, *args, **kwargs):
        style = {**self.STYLE, **kwargs.pop(self.STYLE_KEY, {})}
        kwargs[self.STYLE_KEY] = style
        super().__init__(*args, **kwargs)


class LabeledUpload(dash.html.Div):
    """
    Upload component with in-line labels.
    """

    STYLE = {'display': 'inline-block'}
    STATUS_STYLE = {**STYLE, 'margin-left': '10px', 'margin-right': '5px'}

    def __init__(self,
                 label=None,
                 status_id=None,
                 button_id=None,
                 click_id=None):
        self.label = dash.html.Div(children=label, style=self.STYLE)
        self.status = dash.html.Div(children='',
                                    id=status_id,
                                    style=self.STATUS_STYLE)
        button = Upload(
            id=button_id,
            children=dash.html.Div(children='', id=click_id),
        )
        self.button = dash.html.Div(children=button, style=self.STYLE)
        super().__init__(children=[self.label, self.status, self.button])
