'''Keep configuration settings for mtpy in CFG files.

This module will try to load the ``mtpy.cfg`` settings file from three locations:
the current working directory, the user's home directory, and the original
in the package source directory (SYSTEM_CFG), in that priority. Each file
need not contain all settings, and settings from higher priority files take
precendence.

Usage
-----

For the configuration file containing this text::

    [BIRRP]
    bbfile = Z:\instruments\bbconv.txt

you can access settings like so::
    
    >>> from MTpy.config import config
    >>> for section_name in config.sections():
    ...     print 'Section:', section_name
    ...     for name, value in config.items(section_name):
    ...         print '\t', name, ':', value
    Section: BIRRP
        bbfile : Z:\instruments\bbconv.txt

Format
------
        
The settings file should be in CFG file format. See for more info:

http://www.doughellmann.com/PyMOTW/ConfigParser/#accessing-configuration-settings

Github
------

If for some reason the SYSTEM_CFG file is missing, load will try to download the
original from Github, and write it as a backup 'github-mtpy.cfg' configuration
file.
        
'''
import ConfigParser
import os
import shutil
import urllib2

CFG_FN = 'mtpy.cfg'

# Possible locations for CFG files

GITHUB_CFG = os.path.join(os.path.dirname(__file__), 'github-' + CFG_FN)
SYSTEM_CFG = os.path.join(os.path.dirname(__file__), CFG_FN)
USER_CFG = os.path.join(os.path.expanduser('~'), CFG_FN)
WORKINGDIR_CFG = os.path.join(os.getcwd(), CFG_FN)
    
    
def load():
    '''Return config parser.
    
    Checks for GITHUB_CFG, SYSTEM_CFG, USER_CFG, and WORKINGDIR_CFG and updates
    the parser object with the values found each time.
    
    '''
    def combine(parsers):
        '''Return a new SafeConfigParser containing the options already set in
        *parsers*.
        
        Works forwards through the list, so the higher priority parsers should go
        at the end of *parsers*.
        
        '''
        new_parser = ConfigParser.SafeConfigParser()
        for parser in parsers:
            if not parser:
                continue
            for section in parser.sections():
                if not new_parser.has_section(section):
                    new_parser.add_section(section)
                for name, value in parser.items(section):
                    new_parser.set(section, name, str(value))
        return new_parser
    
    def make_parser(fn):
        parser = ConfigParser.SafeConfigParser()
        parser.read(fn)
        return parser
    
    parser = make_parser(GITHUB_CFG)
    read_at_least_one = False
    for fn in [SYSTEM_CFG, USER_CFG, WORKINGDIR_CFG]:
        next_parser = make_parser(fn)
        if next_parser.read(fn):
            read_at_least_one = True
            parser = combine([parser, next_parser])
            
    if not read_at_least_one:
        # Failed to read any CFG file. This is bad, so try to retrieve the
        # latest default from Github, and copy it to SYSTEM_CFG.
        
        github_cfg_url = urllib2.url('http://raw.github.com/geophysics/mtpy/master/mtpy/' + CFG_FN)
        with open(GITHUB_CFG, mode='w') as f:
            f.write(github_cfg_url.read())
        return parser.read([GITHUB_CFG])
    return parser
    
    
config = load()
