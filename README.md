create\_jacobian
===============

This utility creates source code to calculate the Jacobian matrix analytically for a chemical reaction mechanism.

Usage
-----

create\_jacobian can be run either as an executable or script via Python. To run as an executable, from the command line change to the proper directory, change the file mode to executable, and run:

    chmod +x create_jacobian.py
    ./create_jacobian.py [options]

To run it as a script, change to the appropriate directory and run:

    python create_jacobian.py [options]

The generated source code is placed within the `out` directory, which is created if it doesn't exist initially.

Input reaction mechanisms are supported in Chemkin or Cantera formats. For Cantera, the
Cantera Python package must be installed, and the program looks for files ending with
.cti or .xml.

Options
-------

In the above, `[options]` indicates where command line options should be specified. The options available can be seen using `-h` or `--help`, or below:

    -h, --help            show this help message and exit
    -l {c,cuda}, --lang {c,cuda}
                          Programming language for output source files.
    -i INPUT, --input INPUT
                          Input mechanism filename (e.g., mech.dat).
    -t THERMO, --thermo THERMO
                          Thermodynamic database filename (e.g., therm.dat), or
                          nothing if in mechanism (or Cantera-format).

License
-------

create\_jacobian is released under the modified BSD license, see LICENSE for details.

If you use this package as part of a scholarly publication, please cite the following paper in addition to this resource:

 * TBD

Author
------

Created by [Kyle Niemeyer](http://kyleniemeyer.com). Email address: [kyle.niemeyer@gmail.com](mailto:kyle.niemeyer@gmail.com)
