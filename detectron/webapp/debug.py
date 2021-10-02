from streamlit import bootstrap

real_script = 'predictor.py'

bootstrap.run(real_script, f'debug.py {real_script}', [], {})