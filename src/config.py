import os

# directories
TEMPDIR = os.environ.get('TEMPDIR')
ROOTDIR = os.environ.get('ROOTDIR')
CODEDIR = os.path.join(ROOTDIR, 'code')
DATADIR = os.path.join(ROOTDIR, 'data')
OUTDIR = os.path.join(ROOTDIR, 'output')
FIGUREDIR = os.path.join(OUTDIR, 'figures')
TABLEDIR = os.path.join(OUTDIR, 'tables')
MODELDIR = os.path.join(CODEDIR, 'models')

# globals
TAGVAR = 'up_tag'
