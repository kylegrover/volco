import logging
import json
import time

from app.configs.printer import Printer
from app.configs.simulation import Simulation
from app.configs.arguments import Arguments
from app.geometry.voxel_space import VoxelSpace
from app.instructions.gcode import Gcode
from app.reporter.report import SimulationOutput


logging.basicConfig(format="%(levelname)s %(asctime)s %(message)s", level=logging.INFO)


def run_simulation(
    gcode=None,
    printer_config=None,
    sim_config=None,
    gcode_path=None,
    printer_config_path=None,
    sim_config_path=None,
):
    """
    Run the Volco simulation from a Python environment (e.g., Jupyter notebook).
    
    Parameters:
    -----------
    gcode : str
        G-code content as a string
    printer_config : dict
        Printer configuration as a dictionary
    sim_config : dict
        Simulation configuration as a dictionary
    gcode_path : str
        Path to G-code file (alternative to gcode parameter)
    printer_config_path : str
        Path to printer configuration file (alternative to printer_config parameter)
    sim_config_path : str
        Path to simulation configuration file (alternative to sim_config parameter)
        
    Returns:
    --------
    SimulationOutput
        The simulation output object containing the voxel space and mesh data
        
    Notes:
    ------
    After getting the simulation output object, you can:
    1. Export to STL: output.export_mesh_to_stl()
    2. Visualize with trimesh: scene = output.visualize_mesh()
    3. Visualize with Plotly: fig = output.visualize_mesh(visualizer='plotly')
    4. Access the voxel space directly: output.voxel_space
    
    """
    start_time = time.time()
    
    # Initialize printer configuration
    if printer_config:
        printer = Printer(config_dict=printer_config)
    elif printer_config_path:
        printer = Printer(config_path=printer_config_path)
    else:
        raise ValueError("Either printer_config or printer_config_path must be provided")
    
    # Initialize G-code instruction
    default_nozzle_speed = 50.0  # Default nozzle speed in mm/s
    if gcode:
        instruction = Gcode(gcode_content=gcode, default_nozzle_speed=default_nozzle_speed, printer=printer)
    elif gcode_path:
        instruction = Gcode(gcode_path=gcode_path, default_nozzle_speed=default_nozzle_speed, printer=printer)
    else:
        raise ValueError("Either gcode or gcode_path must be provided")
    
    instruction.read()
    print(f"Number of printed filaments: {instruction.number_printed_filaments}")
    
    # Initialize simulation configuration
    if sim_config:
        simulation = Simulation(config_dict=sim_config, printer=printer)
    elif sim_config_path:
        simulation = Simulation(config_path=sim_config_path, printer=printer)
    else:
        raise ValueError("Either sim_config or sim_config_path must be provided")
    
    # Initialize voxel space
    voxel_space = VoxelSpace(
        instruction=instruction,
        simulation_config=simulation,
        printer=printer,
    )
    
    voxel_space.initialize_space()
    voxel_space.print()
    
    # Process simulation output
    output = SimulationOutput(voxel_space=voxel_space, simulation=simulation)
    
    # Crop the voxel space
    output.crop_voxel_space()

    # Debug: Log stl_ascii setting
    # logger.info(f"[Volco]: stl_ascii setting is {getattr(simulation, 'stl_ascii', 'Not Set')}")

    # Generate mesh
    # We now use streaming export for both binary and ASCII, so we don't need to generate a full mesh object
    logging.info("[Volco]: Skipping mesh generation (using streaming export)")

    print(f"\nTotal simulation time: {time.time() - start_time:.2f} seconds")
    
    return output


if __name__ == "__main__":
    options = Arguments.get_options()

    # Load simulation config from file if provided
    sim_config = None
    if options.sim:
        with open(options.sim, "r") as f:
            sim_config = json.load(f)
    # If --preview is set, override preview_mode in config
    if options.preview:
        if sim_config is None:
            sim_config = {}
        sim_config["preview_mode"] = True

    output = run_simulation(
        gcode_path=options.gcode,
        printer_config_path=options.printer,
        sim_config=sim_config if sim_config is not None else None,
        sim_config_path=None if sim_config is not None else options.sim
    )
    
    # Export STL
    output.export_mesh_to_stl()
