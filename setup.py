from setuptools import setup

setup(
    name='tino_cta',
    version='0.1',
    description='A useful module',
    author='Tino Michael',
    author_email='tino.michael@cea.fr',
    packages=['tino_cta'],
    scripts=[
        'scripts/classify_and_reconstruct.py',
        'scripts/compare_wave_tail_simple.py',
        'scripts/fit_events_hillas.py',
        'scripts/train_classifier.py',
        'scripts/train_energy_regressor.py',
        'scripts/write_feature_table.py'
    ],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'astropy', 'pandas',
                      'matplotlib']
)
