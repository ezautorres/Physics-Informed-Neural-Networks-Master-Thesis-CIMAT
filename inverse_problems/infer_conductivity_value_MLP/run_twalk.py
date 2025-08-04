import sys, os                                                                       # Import sys and os modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))   # Add the parent directory to the path.
import torch
from utils import load_full_model, get_model_info
from inverse_problems.infer_conductivity_value_MLP.infer_conductivity_value_MLP import InferringConductivityValue # Import the InferringConductivityValue class.

checkpoint_filename = "infer_conductivity_value_MLP.pth" # Name of the checkpoint file.

infer_rho_pinn = load_full_model(
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_models", checkpoint_filename),
    model_class     = InferringConductivityValue,
)

get_model_info(checkpoint_filename)        # Print model information.

#x = torch.tensor([[1.0, 0.0, 3.2, 0.85]])  # (1, 4) tensor
#u = infer_rho_pinn.analytical_solution(x)
#print(u.item())  # Para imprimir el escalar
#
## Asegúrate de estar en el mismo dispositivo que el modelo
#x = x.to(next(infer_rho_pinn.pinn.parameters()).device)
#
## Evaluar la PINN
#u_pred = infer_rho_pinn.pinn(x)
#print(u_pred.item())