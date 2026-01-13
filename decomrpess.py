import argparse
import sys
import ReaderClass
import numpy as np
from mpi4py import MPI
import adios2
from rich.traceback import install
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate compressed ADIOS2 files against original")

    parser.add_argument(
        "--original",
        "-orig",
        type=str,
        required=True,
        help="Original uncompressed BP file"
    )
    parser.add_argument(
        "--compressed",
        "-comp",
        type=str,
        required=True,
        help="Compressed BP file (will search for all EB variants)"
    )
    parser.add_argument(
        "--operator",
        "-op",
        type=str,
        required=True,
        help="Operator used (MGARD or CAESAR)"
    )
    parser.add_argument(
        "--xml",
        "-x",
        type=str,
        default=None,
        help="Path to ADIOS2 XML configuration file (optional)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress detailed progress messages"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="validation_results.txt",
        help="Output file for results"
    )

    return parser.parse_args()


def compute_nrmse(original, decompressed):
    """Compute Normalized Root Mean Square Error"""
    mse = np.mean((original - decompressed) ** 2)
    rmse = np.sqrt(mse)
    data_range = np.max(original) - np.min(original)
    if data_range == 0:
        return 0.0
    nrmse = rmse / data_range
    return nrmse


def compute_psnr(original, decompressed):
    """Compute Peak Signal-to-Noise Ratio"""
    mse = np.mean((original - decompressed) ** 2)
    if mse == 0:
        return float('inf')
    max_val = np.max(np.abs(original))
    if max_val == 0:
        return float('inf')
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def compute_compression_ratio(original_file, compressed_file):
    """Compute compression ratio based on file sizes"""
    try:
        # For BP directories, sum all files
        if os.path.isdir(original_file):
            orig_size = sum(
                os.path.getsize(os.path.join(original_file, f))
                for f in os.listdir(original_file)
                if os.path.isfile(os.path.join(original_file, f))
            )
        else:
            orig_size = os.path.getsize(original_file)
        
        if os.path.isdir(compressed_file):
            comp_size = sum(
                os.path.getsize(os.path.join(compressed_file, f))
                for f in os.listdir(compressed_file)
                if os.path.isfile(os.path.join(compressed_file, f))
            )
        else:
            comp_size = os.path.getsize(compressed_file)
        
        if comp_size == 0:
            return 0.0
        return orig_size / comp_size
    except Exception as e:
        print(f"Warning: Could not compute compression ratio: {e}")
        return 0.0


def main():
    install()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size > 1:
        print("This code works single rank")
        sys.exit()

    args = parse_arguments()

    original_file = args.original
    compressed_base = args.compressed
    op_name = args.operator.lower()
    xml = args.xml
    quiet = args.quiet
    output_file = args.output

    # Error bounds to test
    error_bounds = [1e-2, 1e-3, 1e-4, 1e-5]

    print(f"Starting validation for {op_name.upper()}")
    print(f"Original file: {original_file}")
    print(f"Results will be saved to: {output_file}")

    results = []
    results.append("="*80)
    results.append(f"VALIDATION RESULTS - {op_name.upper()}")
    results.append(f"Original file: {original_file}")
    results.append("="*80)
    results.append("")

    overall_pass = True

    for eb_idx, eb in enumerate(error_bounds):
        eb_str = f"{eb:.0e}".replace("e-0", "e-")
        compressed_file = compressed_base.replace(".bp", f"_compressed_{op_name}_eb_{eb_str}.bp")

        print(f"\n{'='*60}")
        print(f"Validating Error Bound: {eb}")
        print(f"Compressed file: {compressed_file}")
        print(f"{'='*60}")

        results.append(f"\nError Bound: {eb}")
        results.append("-"*60)

        if not os.path.exists(compressed_file):
            msg = f"✗ FAIL: Compressed file not found: {compressed_file}"
            print(msg)
            results.append(msg)
            results.append("")
            overall_pass = False
            continue

        try:
            # Compute compression ratio
            cr = compute_compression_ratio(original_file, compressed_file)
            results.append(f"Compression Ratio: {cr:.2f}x")
            print(f"Compression Ratio: {cr:.2f}x")

            # Initialize readers with unique IO names for each error bound
            unique_reader_orig = f"reader_orig_eb{eb_idx}"
            unique_reader_comp = f"reader_comp_eb{eb_idx}"
            
            r_orig = ReaderClass.Reader(unique_reader_orig, original_file, xml=xml, comm=comm)
            r_comp = ReaderClass.Reader(unique_reader_comp, compressed_file, xml=xml, comm=comm)

            step_count = 0
            all_vars_pass = True
            failed_vars = []
            var_results = []

            while True:
                status_orig = r_orig.begin_step()
                status_comp = r_comp.begin_step()

                if status_orig != adios2.bindings.StepStatus.OK or \
                   status_comp != adios2.bindings.StepStatus.OK:
                    break

                current_step = r_orig.current_step()
                if not quiet:
                    print(f"  Validating step {current_step}")

                available_vars = r_orig.Adios_reader.available_variables()

                for var_name, var_info in available_vars.items():
                    var_type = var_info.get('Type', '')

                    if var_type in ['double', 'float']:
                        # Read original data
                        r_orig.set_read_vars([var_name])
                        data_orig = r_orig.read_step(var_name)

                        # Read decompressed data
                        r_comp.set_read_vars([var_name])
                        data_comp = r_comp.read_step(var_name)

                        # Ensure same dtype
                        if data_orig.dtype != np.float64:
                            data_orig = np.array(data_orig, dtype=np.float64)
                        if data_comp.dtype != np.float64:
                            data_comp = np.array(data_comp, dtype=np.float64)

                        # Compute metrics
                        nrmse = compute_nrmse(data_orig, data_comp)
                        psnr = compute_psnr(data_orig, data_comp)

                        # Check if NRMSE is within error bound
                        passed = nrmse <= eb

                        if not quiet:
                            status_symbol = "✓ PASS" if passed else "✗ FAIL"
                            print(f"    {status_symbol} {var_name}: NRMSE={nrmse:.6e}, PSNR={psnr:.2f} dB")

                        var_results.append({
                            'name': var_name,
                            'step': current_step,
                            'nrmse': nrmse,
                            'psnr': psnr,
                            'passed': passed
                        })

                        if not passed:
                            all_vars_pass = False
                            failed_vars.append(f"{var_name} (step {current_step}): NRMSE={nrmse:.6e} > {eb}")

                r_orig.end_step()
                r_comp.end_step()
                step_count += 1

            # Properly close readers
            r_orig.close()
            r_comp.close()
            
            # Clean up by deleting readers
            del r_orig
            del r_comp

            # Summarize results for this error bound
            results.append(f"Total steps validated: {step_count}")
            results.append(f"Total variables checked: {len(var_results)}")

            # Calculate statistics
            if var_results:
                avg_nrmse = np.mean([v['nrmse'] for v in var_results])
                max_nrmse = np.max([v['nrmse'] for v in var_results])
                avg_psnr = np.mean([v['psnr'] for v in var_results if np.isfinite(v['psnr'])])

                results.append(f"Average NRMSE: {avg_nrmse:.6e}")
                results.append(f"Maximum NRMSE: {max_nrmse:.6e}")
                results.append(f"Average PSNR: {avg_psnr:.2f} dB")

            if all_vars_pass:
                msg = f"✓ PASS: All variables meet error bound {eb}"
                print(f"\n{msg}")
                results.append(f"\n{msg}")
            else:
                msg = f"✗ FAIL: {len(failed_vars)} variable(s) exceed error bound {eb}"
                print(f"\n{msg}")
                results.append(f"\n{msg}")
                results.append("\nFailed variables:")
                for fail in failed_vars:
                    results.append(f"  - {fail}")
                overall_pass = False

            results.append("")

        except Exception as e:
            msg = f"✗ ERROR during validation: {e}"
            print(msg)
            results.append(msg)
            results.append("")
            import traceback
            traceback.print_exc()
            overall_pass = False
            continue

    # Final summary
    results.append("="*80)
    if overall_pass:
        final_msg = "✓ OVERALL: PASSED - All error bounds meet requirements"
    else:
        final_msg = "✗ OVERALL: FAILED - Some error bounds do not meet requirements"
    results.append(final_msg)
    results.append("="*80)

    print(f"\n{'='*60}")
    print(final_msg)
    print(f"{'='*60}")

    # Write results to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(results))

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
