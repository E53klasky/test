import argparse
import sys
import ReaderClass
import WrighterClass
import numpy as np
from mpi4py import MPI
import time
from rich.traceback import install
import adios2


def parse_arguments():
    parser = argparse.ArgumentParser(description="copier code for to get all doubles and floats")
    
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        required=False,
        help="IO Name for the first Adios file (optional)",
    )
    parser.add_argument(
        "--WrightIO",
        "-wio",
        type=str,
        default="writer1",
        required=False,
        help="IO Name for the second Adios file (optional)",
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        help="Path to ADIOS2 XML configuration file (optional)",
    )
    parser.add_argument(
        "--operator",
        "-op",
        type=str,
        default=None,
        required=True,
        help="operator you want to test ex (CAESAR or MGARD)"
    )

    return parser.parse_args()


def main():
    install()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1:
        print("This code works single rank I don't know how to do some of this in parallel")
        sys.exit()

    args = parse_arguments()
    datasets = ['TGV-Taylor-Green-vortex.65.65.65.bp5']
    
    # Get parsed arguments
    readIO = args.readIO
    wrightIO = args.WrightIO
    xml = args.xml
    op = args.operator
    
    print(f"Starting test with {op} as operator")

    current_dataset = None
    current_step = None
    current_name = None
    
    try:
        for dataset in datasets:
            current_dataset = dataset
            print(f"On dataset {dataset}")
            new_dataset = dataset.replace(".bp", "")
            output = new_dataset + "OG.bp"
            
            r = ReaderClass.Reader(readIO, dataset, xml=xml, comm=comm)
            w = WrighterClass.Writer(wrightIO, output, xml=xml, comm=comm)

            while True:
                status = r.begin_step()
                if status != adios2.bindings.StepStatus.OK:
                    break

                current_step = r.current_step()
                print(f"On step {current_step}")
                w.begin_step()

                available_vars = r.Adios_reader.available_variables()
                
                for var_name, var_info in available_vars.items():
                    current_name = var_name
                    
                    adios_var = r.Read_IO.inquire_variable(var_name)
                    
                    if adios_var is None:
                        print(f"Could not inquire variable {var_name}")
                        continue
                    
                    var_type = var_info.get('Type', '')
                    
                    print(f"Variable: {var_name}, Type: {var_type}")
                    
                    if var_type in ['double', 'float']:
                        print(f"Reading variable {var_name}")
                        r.set_read_vars([var_name])
                        
                        data = r.read_step(var_name)
                        
                        if data.dtype != np.float64:
                            data = np.array(data, dtype=np.float64)
                        
                        w.write(var_name, data)
                        print(f"Wrote variable {var_name}")
                    else:
                        print(f"NOT a float or double: {var_type}")

                r.end_step()
                w.end_step()
                
            r.close()
            w.close()
            print(f"Done with dataset {dataset} and finished writing {output}")

    except Exception as e:
        print(f"Something went wrong on dataset {current_dataset}, step {current_step}, variable {current_name}, operator {op}")
        print(f"The error was: {e}")
        import traceback
        traceback.print_exc()

    print("Done testing")


if __name__ == "__main__":
    main()
