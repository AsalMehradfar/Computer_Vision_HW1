import configparser


def path_config():
    config = configparser.ConfigParser()
    config.read('Config/path_config')
    path = config['path']
    return path


def param_config(path):
    config = configparser.ConfigParser()
    config.read('Config/' + path)
    params_parser = config['int_params']
    params = {}
    for key in params_parser:
        params[key] = int(params_parser[key])

    params_parser = config['float_params']
    for key in params_parser:
        params[key] = float(params_parser[key])

    return params
