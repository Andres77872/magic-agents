from jinja2 import Template


def template_parse(template, params):
    t = Template(template)
    o = t.render(params)
    return o
