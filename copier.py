import argparse
import sys
import ReaderClass
import WrighterClass
import numpy as np
from mpi4py import MPI
import time
from rich.traceback import install


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
        help= "operator you want to test ex (CAESAR or MGARD)"
    )

    return parser.parse_args()


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1:
        print("This code works single rank I don't know how to do some of this in parallel")
        sys.exit()

    parser = parse_arguments()
    datasets = ['summit.20220527.bp']
    
    print("Starting test with {paraser.operator} as operator")

    # add time out for the ones which have a last bad step not closed maybe maybe not must be tested
    try:
        for dataset in datasets:
            print("On dataset {dataset}")
            new_dataset = dataset.replace(".bp", "")
            output = new_dataset + "C.bp"
            r = ReaderClass.Reader( readIO, dataset, xml=xml, comm=comm)
            w = WrighterClass.Writer( wrightIO, output, xml=xml, comm=comm)

            while True:
                status = r.begin_step()
                if status != adios2.bindings.StepStatus.OK:
                    break

                current_step = r.current_step()
                print("On step {current_step}")
                w.begin_step()

                for name, info in r.Adios_reader.available_variables().items():
                    name.info()
                    print(info)
                    print(name.info)
                    data_typ = np.dtype(name)
                    if data_typ == np.float64 or data_typ == np.float32:
                        r.set_read_vars([name])

                        data = r.read_step(name)
                        print("Rading varible {name}")
                        data = np.array(r.read_step(name), dtype=np.float64)

                        w.write(name, data)
                        print("Wrote varible {name}")
                    else:
                        print("NOT a float or double")


                r.end_step()
                w.end_step()
            r.close()
            w.close()
            print("Done with dataset {dataset} and finshed wrighting {output}")


    except Exception as e:
        print("something went wrong on dataset {dataset}, step {step}, varible {name}, operator {op}")
        print("The error was {e}")

    print("Done testing")

if __name__ == "__main__":
    install()
    main()
