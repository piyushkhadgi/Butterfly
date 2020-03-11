
def import_or_install(packages):
    import pip
    for package in packages:
        try:
            __import__(package)
        except ImportError:
            pip.main(['install', package])


