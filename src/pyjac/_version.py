__version__ = '1.0.6'
__version_info__ = tuple(
    int(part) if part.isdigit() else part
    for part in __version__.replace('-', '.').split('.')
)
