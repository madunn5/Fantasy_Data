from django import template

register = template.Library()

@register.filter
def get_range(value):
    """
    Generate a range of numbers.
    Usage: {% for i in balls|get_range %}...{% endfor %}
    """
    return range(int(value))