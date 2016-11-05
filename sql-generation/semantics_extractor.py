#!/usr/bin/env python

import sys
import re
import collections
import numpy as np
import h5py


def extract_tables_and_columns(semantics_dict, sql_dump):
    """
    :p semantics: dictionary that will contain the table names
    :t semantics: dict

    :p sql_tables: file form which we read lines
    :t sql_tables: file

    returns semantics dictionary populated with table
    and columns
    """
    table = None
    for line in sql_dump:
        table_name = re.match("CREATE TABLE (\w+)", line)
        #print(table_name, "!")
        if table_name is not None:
            table = table_name.group(1).replace("`", "").strip()
            print("New Table:")
            print(table)
            semantics_dict[table] = collections.OrderedDict()
            semantics_dict[table]["num_of_accesses"] = 0
        elif table is not None:
            # Note: this is table not table_name, like above
            column = re.match("  (\w+) int", line)
            #print(column)
            if column is not None:
                key_2 = column.group(1).replace("`", "").strip()
                print(key_2)
                (semantics_dict[table])[key_2] = 0
            else:  # inside a table but nothing is declared? Exit
                end_paren = re.match("\)", line)
                if end_paren is not None:
                    table = None
                #else:
                    #print("Something is wrong with the SQL file")
                    #sys.exit(1)
        else:
            #  table_name is None and table is None
            continue
    return semantics_dict


# ==============================================
# main
# ==============================================
if __name__ == '__main__':
    ###################
    # Error checking  #
    ###################
    if ".sql" not in sys.argv[1]:
        print("Error: Need a sql file as the first argument to python file")
        sys.exit(1)

    sql_file = sys.argv[1]

    ##############################
    # Extracting files to fileIO #
    ##############################

    try:
        sql_file = open(sql_file, 'r', encoding='utf-8')
    except:
        print("Specified sql table format file does not exist")
        sys.exit(1)

    ###############################
    # # Build semantic dictionary #
    ###############################

    table_dict = collections.OrderedDict()
    semantics_dict = extract_tables_and_columns(table_dict, sql_file)
    print("Finished building table for sql tree dump")


