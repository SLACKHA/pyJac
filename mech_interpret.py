from chem_utilities import *
import utils

def read_mech(filename, elems, specs, reacs):
    """Read and interpret mechanism file for elements, species, and reactions.
    
    Doesn't support element names with digits.
    
    Input
    filename:  reaction mechanism filename (e.g. 'mech.dat')
    """
    
    file = open(filename, 'r')
    
    num_e = 0
    num_s = 0
    num_r = 0
    
    units = ''
    key = ''
    
    # start line reading loop
    while True:
        # remember last line position
        last_line = file.tell()
        
        line = file.readline()
        
        # end of file
        if not line: break
        
        # skip blank or commented lines
        if line == '\n' or line == '\r\n' or line[0:1] == '!': continue
        
        # don't convert everything, since thermo needs to match (for Chemkin)
        ## convert to lowercase
        #line = line.lower()
        
        # remove any comments from end of line
        ind = line.find('!')
        if ind > 0: line = line[0:ind]
        
        # now determine key
        if line[0:4].lower() == 'elem':
            key = 'elem'
            
            # check for any entries on this line
            line_split = line.split()
            if len(line_split) > 1:
                ind = line.index( line_split[1] )
                line = line[ind:]
            else:
                continue
            
        elif line[0:4].lower() == 'spec':
            key = 'spec'
            
            # check for any entries on this line
            line_split = line.split()
            if len(line_split) > 1:
                ind = line.index( line_split[1] )
                line = line[ind:]
            else:
                continue
            
        elif line[0:4].lower() == 'reac':
            key = 'reac'
            
            # get Arrhenius coefficient units
            line_split = line.split()
            if len(line_split) > 1:
                units = line[ line.index(line_split[1]) : ].strip()
            else:
                # default units
                units = 'cal/mole'
            
            continue
        elif line[0:4].lower() == 'ther':
            # thermo data is in mechanism file
            
            # rewind a line
            file.seek(last_line)
            
            read_thermo(file, elems, specs)
            
            continue
        elif line[0:3].lower() == 'end':
            key = ''
            continue
        
        line = line.strip()
        
        if key == 'elem':
            # if any atomic weight declarations, replace / with spaces
            line = line.replace('/', ' ')
            
            line_split = line.split()
            e_last = ''
            for e in line_split:
                if e.isalpha():
                    if e[0:3] == 'end': continue
                    if e not in elems:
                        elems.append(e)
                        num_e += 1
                    e_last = e
                else:
                    # check either new element or updating existing atomic weight
                    if e_last in elem_mw:
                        elem_mw[e_last.lower()] = float(e)
                    
                    # in both cases add to 2nd dict to keep track
                    elem_mw_new[e_last.lower()] = float(e)
            
        elif key == 'spec':
            line_split = line.split()
            for s in line_split:
                if s[0:3] == 'end': continue
                if not next((sp for sp in specs if sp.name == s), None):
                    specs.append( SpecInfo(s) )
                    num_s += 1
            
        elif key == 'reac':
            # determine if reaction or auxiliary info line
            
            if '=' in line:
                # new reaction
                num_r += 1
                
                # get Arrhenius coefficients
                line_split = line.split()
                n = len(line_split)
                reac_A = float( line_split[n - 3] )
                reac_b = float( line_split[n - 2] )
                reac_E = float( line_split[n - 1] )
                
                ind = line.index( line_split[n - 3] )
                line = line[0:ind].strip()
                
                if '<=>' in line:
                    ind = line.index('<=>')
                    reac_rev = True
                    reac_str = line[0:ind].strip()
                    prod_str = line[ind + 3:].strip()
                elif '=>' in line:
                    ind = line.index('=>')
                    reac_rev = False
                    reac_str = line[0:ind].strip()
                    prod_str = line[ind + 2:].strip()
                else:
                    ind = line.index('=')
                    reac_rev = True
                    reac_str = line[0:ind].strip()
                    prod_str = line[ind + 1:].strip()
                
                thd = False
                pdep = False
                pdep_sp = ''
                
                reac_spec = []
                reac_nu = []
                prod_spec = []
                prod_nu = []
                
                # reactants
                
                # look for third-body species
                sub_str = reac_str
                while '(' in sub_str:
                    ind1 = sub_str.find('(')
                    ind2 = sub_str.find(')')
                    
                    # need to check if '+' is first character inside parentheses
                    # and not embedded within parentheses (e.g., '(+)')
                    # if not, part of species name
                    inParen = sub_str[ind1 + 1 : ind2].strip()
                    if inParen is '+':
                        # '+' embedded within parentheses
                        sub_str = sub_str[ind2 + 1:]
                    elif inParen[0] is '+':
                        pdep = True
                        
                        # either 'm' or a specific species
                        pdep_sp = sub_str[ind1 + 1 : ind2].replace('+', ' ')
                        pdep_sp = pdep_sp.strip()
                        
                        if pdep_sp.lower() == 'm':
                            thd = True
                            pdep_sp = ''
                    
                        # now remove from string
                        ind = reac_str.find(sub_str)
                        reac_str = reac_str[0 : ind1 + ind] + reac_str[ind2 + ind + 1 :]
                        break
                    else:
                        # part of species name, remove from substring and look at rest of reactant line
                        sub_str = sub_str[ind2 + 1:]
                
                reac_list = reac_str.split('+')
                
                # check for empty list elements, meaning there were multiple '+' in a row
                # indicates species name ended in '+'
                while '' in reac_list:
                    ind = reac_list.index('')
                    reac_list[ind - 1] = reac_list[ind - 1] + '+'
                    del reac_list[ind]
                
                # check for any species with '(+)' that was split apart
                for sp in reac_list:
                    ind = reac_list.index(sp)
                    
                    # ensure not last entry
                    if (ind < len(reac_list) - 1):
                        spNext = reac_list[ind + 1]
                        if sp[len(sp) - 1] is '(' and spNext[0] is ')':
                            reac_list[ind] = sp + '+' + spNext
                            del reac_list[ind + 1]
                
                for sp in reac_list:
                    
                    sp = sp.strip()
                    
                    # look for coefficient
                    if sp[0:1].isdigit(): 
                        # starts with number (coefficient)
                        
                        # search for first letter
                        for i in range( len(sp) ):
                            if sp[i : i + 1].isalpha(): break
                        
                        nu = sp[0:i]
                        if '.' in nu:
                            # float
                            nu = float(nu)
                        else:
                            # integer
                            nu = int(nu)
                        
                        sp = sp[i:].strip()
                    else:
                        # no coefficient given
                        nu = 1
                    
                    # check for third body
                    if sp.lower() == 'm':
                        thd = True
                        continue
                    
                    # check if species already in reaction
                    if sp not in reac_spec:
                        # new reactant
                        reac_spec.append(sp)
                        reac_nu.append(nu)
                    else:
                        # existing reactant
                        i = reac_spec.index(sp)
                        reac_nu[i] += nu
                
                # products
                
                # look for third-body species
                sub_str = prod_str
                while '(' in sub_str:
                    ind1 = sub_str.find('(')
                    ind2 = sub_str.find(')')

                    # need to check if '+' is first character inside parentheses
                    # and not embedded within parentheses (e.g., '(+)')
                    # if not, part of species name
                    inParen = sub_str[ind1 + 1 : ind2].strip()
                    if inParen is '+':
                        # '+' embedded within parentheses
                        sub_str = sub_str[ind2 + 1:]
                    elif inParen[0] is '+':
                        pdep = True

                        # either 'm' or a specific species
                        pdep_sp = sub_str[ind1 + 1 : ind2].replace('+', ' ')
                        pdep_sp = pdep_sp.strip()

                        if pdep_sp.lower() == 'm':
                            thd = True
                            pdep_sp = ''

                        # now remove from string
                        ind = prod_str.find(sub_str)
                        prod_str = prod_str[0 : ind1 + ind] + prod_str[ind2 + ind + 1 :]
                        break
                    else:
                        # part of species name, remove from substring and look at rest of product line
                        sub_str = sub_str[ind2 + 1:]
                
                prod_list = prod_str.split('+')
                
                # check for empty list elements, meaning there were multiple '+' in a row
                # indicates species name ended in '+'
                while '' in prod_list:
                    ind = prod_list.index('')
                    prod_list[ind - 1] = prod_list[ind - 1] + '+'
                    del prod_list[ind]
                
                # check for any species with '(+)' that was split apart
                for sp in prod_list:
                    ind = prod_list.index(sp)

                    # ensure not last entry
                    if (ind < len(prod_list) - 1):
                        spNext = prod_list[ind + 1]
                        if sp[len(sp) - 1] is '(' and spNext[0] is ')':
                            prod_list[ind] = sp + '+' + spNext
                            del prod_list[ind + 1]
                
                for sp in prod_list:
                    
                    sp = sp.strip()
                    
                    # look for coefficient
                    if sp[0:1].isdigit(): 
                        # starts with number (coefficient)
                        
                        # search for first letter
                        for i in range( len(sp) ):
                            if sp[i : i + 1].isalpha(): break
                        
                        nu = sp[0:i]
                        if '.' in nu:
                            # float
                            nu = float(nu)
                        else:
                            # integer
                            nu = int(nu)
                        
                        sp = sp[i:].strip()
                    else:
                        # no coefficient given
                        nu = 1
                    
                    # check for third body
                    if sp == 'm' or sp == 'M':
                        thd = True
                        continue
                    
                    # check if species already in reaction
                    if sp not in prod_spec:
                        # new product
                        prod_spec.append(sp)
                        prod_nu.append(nu)
                    else:
                        # existing product
                        i = prod_spec.index(sp)
                        prod_nu[i] += nu
                
                # add reaction to list
                reacs.append( ReacInfo(reac_rev, reac_spec, reac_nu, prod_spec, prod_nu, reac_A, reac_b, reac_E) )
                reacs[num_r - 1].thd = thd
                reacs[num_r - 1].pdep = pdep
                if pdep: reacs[num_r - 1].pdep_sp = pdep_sp
                
            else:
                # auxiliary reaction info
                
                aux = line[0:3].lower()
                if aux == 'dup':
                    reacs[num_r - 1].dup = True
                    
                elif aux == 'rev':
                    line = line.replace('/', ' ')
                    line = line.replace(',', ' ')
                    line_split = line.split()
                    reacs[num_r - 1].rev_par.append( float( line_split[1] ) )
                    reacs[num_r - 1].rev_par.append( float( line_split[2] ) )
                    reacs[num_r - 1].rev_par.append( float( line_split[3] ) )
                    
                elif aux == 'low':
                    line = line.replace('/', ' ')
                    line = line.replace(',', ' ')
                    line_split = line.split()
                    reacs[num_r - 1].low.append( float( line_split[1] ) )
                    reacs[num_r - 1].low.append( float( line_split[2] ) )
                    reacs[num_r - 1].low.append( float( line_split[3] ) )
                    
                elif aux == 'hig':
                    line = line.replace('/', ' ')
                    line = line.replace(',', ' ')
                    line_split = line.split()
                    reacs[num_r - 1].high.append( float( line_split[1] ) )
                    reacs[num_r - 1].high.append( float( line_split[2] ) )
                    reacs[num_r - 1].high.append( float( line_split[3] ) )
                    
                elif aux == 'tro':
                    line = line.replace('/', ' ')
                    line = line.replace(',', ' ')
                    line_split = line.split()
                    reacs[num_r - 1].troe = True
                    reacs[num_r - 1].troe_par.append( float( line_split[1] ) )
                    reacs[num_r - 1].troe_par.append( float( line_split[2] ) )
                    reacs[num_r - 1].troe_par.append( float( line_split[3] ) )
                    
                    # optional fourth parameter
                    if len(line_split) > 4:
                        reacs[num_r - 1].troe_par.append( float( line_split[4] ) )
                    
                elif aux == 'sri':
                    line = line.replace('/', ' ')
                    line = line.replace(',', ' ')
                    line_split = line.split()
                    reacs[num_r - 1].sri = True
                    reacs[num_r - 1].sri_par.append( float( line_split[1] ) )
                    reacs[num_r - 1].sri_par.append( float( line_split[2] ) )
                    reacs[num_r - 1].sri_par.append( float( line_split[3] ) )
                    
                    # optional fourth and fifth parameters
                    if len(line_split) > 4:
                        reacs[num_r - 1].sri_par.append( float( line_split[4] ) )
                        reacs[num_r - 1].sri_par.append( float( line_split[5] ) )
                else:
                    # enhanced third body efficiencies
                    line = line.replace('/', ' ')
                    line_split = line.split()
                    for i in range(0, len(line_split), 2):
                        reacs[num_r - 1].thd_body.append( [line_split[i], float(line_split[i + 1])] )
    
    return (num_e, num_s, num_r, units)


def read_thermo(file, elems, specs):
    """Read and interpret thermodynamic database for species data.
    
    Reads the file therm.dat and returns the species thermodynamic coefficients
    as well as the species-specific temperature range values (if given)
    
    Input
    file:  pointer to open thermo database file
    elems: list of element names
    specs: list of species names (SpecInfo class)
    """
    
    # loop through intro lines
    while True:
        line = file.readline()
    
        # skip blank or commented lines
        if line == '\n' or line == '\r\n' or line[0:1] == '!': continue
    
        # skip 'thermo' at beginning
        if line[0:6].lower() == 'thermo': break
    
    # next line either has common temperature ranges or first species
    last_line = file.tell()
    line = file.readline()
    
    line_split = line.split()
    if line_split[0][0:1].isdigit():
        T_ranges = utils.read_str_num(line)
    else:
        # no common temperature info
        file.seek(last_line)
        # default
        T_ranges = [300.0, 1000.0, 5000,0]
    
    # now start reading species thermo info
    while True:
        # first line of species info
        line = file.readline()
        
        # don't convert to lowercase, needs to match thermo for Chemkin
        #line = line.lower()
        
        # break if end of file
        if line is None: break
        if line[0:3] == 'end': break
        # skip blank/commented line
        if line == '\n' or line == '\r\n' or line[0:1] == '!': continue
        
        # species name, columns 0:18
        spec = line[0:18].strip()
        
        # apparently in some cases notes are in the columns of shorter species names
        # so make sure no spaces
        if spec.find(' ') > 0:
            spec = spec[0 : spec.find(' ')]
        
        # now need to determine if this species is in mechanism
        if next((sp for sp in specs if sp.name == spec), None):
            sp_ind = next(i for i in xrange(len(specs)) if specs[i].name == spec)
        else:
            # not in mechanism, read next three lines and continue
            line = file.readline()
            line = file.readline()
            line = file.readline()
            continue
        
        # set species to the one matched
        spec = specs[sp_ind]
        
        # ensure not reading the same species more than once...
        if spec.mw:
            # already done! skip next three lines
            line = file.readline()
            line = file.readline()
            line = file.readline()
            continue
        
        # now get element composition of species, columns 24:44
        # each piece of data is 5 characters long (2 for element, 3 for #)
        elem_str = utils.split_str(line[24:44], 5)
        
        for e_str in elem_str:
            e = e_str[0:2].strip()
            # skip if blank
            if e == '' or e == '0': continue
            # may need to convert to float first, in case of e.g. "1."
            e_num = float( e_str[2:].strip() )
            e_num = int(e_num)
            
            spec.elem.append([e, e_num])
            
            # calculate molecular weight
            spec.mw += e_num * elem_mw[e.lower()]
        
        # temperatures for species
        T_spec = utils.read_str_num(line[45:73])
        T_low  = T_spec[0]
        T_high = T_spec[1]
        if len(T_spec) == 3: T_com = T_spec[2]
        else: T_com = T_ranges[1]
        
        spec.Trange = [T_low, T_com, T_high]
        
        # second species line
        line = file.readline()
        coeffs = utils.split_str(line[0:75], 15)
        spec.hi[0] = float( coeffs[0] )
        spec.hi[1] = float( coeffs[1] )
        spec.hi[2] = float( coeffs[2] )
        spec.hi[3] = float( coeffs[3] )
        spec.hi[4] = float( coeffs[4] )
        
        # third species line
        line = file.readline()
        coeffs = utils.split_str(line[0:75], 15)
        spec.hi[5] = float( coeffs[0] )
        spec.hi[6] = float( coeffs[1] )
        spec.lo[0] = float( coeffs[2] )
        spec.lo[1] = float( coeffs[3] )
        spec.lo[2] = float( coeffs[4] )
        
        # fourth species line
        line = file.readline()
        coeffs = utils.split_str(line[0:75], 15)
        spec.lo[3] = float( coeffs[0] )
        spec.lo[4] = float( coeffs[1] )
        spec.lo[5] = float( coeffs[2] )
        spec.lo[6] = float( coeffs[3] )
        
        # stop reading if all species in mechanism accounted for
        if not next((sp for sp in specs if sp.mw == 0.0), None): break
    
    return

