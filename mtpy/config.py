'''Keep configuration settings for mtpy in INI files.

This module will try to load the ``mtpy.ini`` settings file from three locations:
the current working directory, the user's home directory, and the original
in the package source directory (SYSTEM_INI), in that priority. Each file
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
        
The settings file should be in INI file format. See for more info:

http://www.doughellmann.com/PyMOTW/ConfigParser/#accessing-configuration-settings

Github
------

If for some reason the SYSTEM_INI file is missing, load will try to download the
original from Github, and write it as a backup 'github-mtpy.ini' configuration
file.
        
'''
import ConfigParser
import os
import shutil
import urllib2

INI_FN = 'mtpy.ini'

# Possible locations for INI files

GITHUB_INI = os.path.join(os.path.dirname(__file__), 'github-' + INI_FN)
SYSTEM_INI = os.path.join(os.path.dirname(__file__), INI_FN)
USER_INI = os.path.join(os.path.expanduser('~'), INI_FN)
WORKINGDIR_INI = os.path.join(os.getcwd(), INI_FN)
    
    
def load():
    '''Return config parser.
    
    Checks for GITHUB_INI, SYSTEM_INI, USER_INI, and WORKINGDIR_INI and updates
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
    
    parser = make_parser(GITHUB_INI)
    read_at_least_one = False
    for fn in [SYSTEM_INI, USER_INI, WORKINGDIR_INI]:
        next_parser = make_parser(fn)
        if next_parser.read(fn):
            read_at_least_one = True
            parser = combine([parser, next_parser])
            
    if not read_at_least_one:
        # Failed to read any INI file. This is bad, so try to retrieve the
        # latest default from Github, and copy it to SYSTEM_INI.
        
        github_ini_url = urllib2.url('http://raw.github.com/geophysics/mtpy/master/mtpy/' + INI_FN)
        with open(GITHUB_INI, mode='w') as f:
            f.write(github_ini_url.read())
        return parser.read([GITHUB_INI])
    return parser
    
    
config = load()
