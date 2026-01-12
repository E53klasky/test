import argparse
import sys
import ReaderClass
import WrighterClass
import numpy as np
from mpi4py import MPI
import adios2
from rich.traceback import install


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compress ADIOS2 files with specified error bounds")
    
    parser.add_argument(
        "--readIO",
        "-rio",
        type=str,
        default="reader1",
        required=False,
        help="IO Name for reading",
    )
    parser.add_argument(
        "--writeIO",
        "-wio",
        type=str,
        default="writer1",
        required=False,
        help="IO Name for writing",
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
        default="mgard",
        required=True,
        help="Compression operator (MGARD or CAESAR)"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input BP file to compress"
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress timestep progress messages"
    )

    return parser.parse_args()


def main():
    install()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1:
        print("This code works single rank")
        sys.exit()

    args = parse_arguments()
    
    readIO = args.readIO
    writeIO = args.writeIO
    xml = args.xml
    op_name = args.operator.lower()
    input_file = args.input
    quiet = args.quiet
    
    # Error bounds to test
    error_bounds = [1e-2, 1e-3, 1e-4, 1e-5]
    
    print(f"Starting compression with {op_name.upper()} operator")
    print(f"Input file: {input_file}")
    print(f"Error bounds: {error_bounds}")
    
    # Verify operator is available
    # For MGARD: operator name is "mgard", parameter is "accuracy"
    # For CAESAR: operator name is "caesar", parameter is "accuracy"
    print(f"Note: Using operator '{op_name}' with 'accuracy' parameter")
    
    for eb in error_bounds:
        eb_str = f"{eb:.0e}".replace("e-0", "e-")
        output_file = input_file.replace(".bp", f"_compressed_{op_name}_eb_{eb_str}.bp")
        
        print(f"\n{'='*60}")
        print(f"Compressing with error bound: {eb}")
        print(f"Output: {output_file}")
        print(f"{'='*60}")
        
        try:
            # Initialize reader and writer
            r = ReaderClass.Reader(readIO, input_file, xml=xml, comm=comm)
            
            # Create ADIOS object for writer with operator
            if xml is not None:
                adios_obj = adios2.Adios(xml, comm=comm)
            else:
                adios_obj = adios2.Adios(comm=comm)
            
            write_io = adios_obj.declare_io(writeIO)
            
            # Open writer
            writer = adios2.Stream(write_io, output_file, "w", comm=comm)
            
            # Define the compression operator via Stream._adios
            compress_op = writer._adios.define_operator(f"{op_name}_op", op_name.lower())
            
            step_count = 0
            
            while True:
                status = r.begin_step()
                if status != adios2.bindings.StepStatus.OK:
                    break
                
                current_step = r.current_step()
                if not quiet:
                    print(f"  Processing step {current_step}")
                
                writer.begin_step()
                
                available_vars = r.Adios_reader.available_variables()
                
                for var_name, var_info in available_vars.items():
                    var_type = var_info.get('Type', '')
                    
                    if var_type in ['double', 'float']:
                        # Read the variable
                        r.set_read_vars([var_name])
                        data = r.read_step(var_name)
                        
                        if data.dtype != np.float64:
                            data = np.array(data, dtype=np.float64)
                        
                        # Define variable in writer with compression
                        shape = list(data.shape)
                        var = write_io.define_variable(
                            var_name,
                            data,
                            shape,
                            [0] * len(shape),
                            shape,
                            is_constant_dims=False
                        )
                        
                        # Add compression operation
                        if op_name == "mgard":
                            var.add_operation(compress_op, {"tolerance": str(eb), "mode": "REL"})
                        else:  # caesar
                            var.add_operation(compress_op, {"batch_size": str(128), "error_bound": str(eb)})
                        
                        writer.write(var_name, data)
                        
                        if not quiet:
                            print(f"    Compressed: {var_name}")
                
                r.end_step()
                writer.end_step()
                step_count += 1
            
            r.close()
            writer.close()
            
            print(f"✓ Successfully compressed {step_count} steps with EB={eb}")
            print(f"  Output: {output_file}")
            
        except Exception as e:
            print(f"✗ Error compressing with EB={eb}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Compression complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
