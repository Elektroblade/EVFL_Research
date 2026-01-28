import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import flwr.client as fl

from vertical_fl.task import (
    load_client_data,
    ClientModel,
    CLIENT_FEATURE_MAP,
)

# ------------------------------------------------------------
# FLOWER VFL CLIENT
# ------------------------------------------------------------

class FlowerVFLClient(fl.client.Client):
    def __init__(self, cid: str, full_df):
        self.cid = int(cid)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---------------------------
        # Load client-specific data
        # ---------------------------
        X = load_client_data(full_df, self.cid)
        dataset = TensorDataset(X)
        self.loader = DataLoader(dataset, batch_size=256, shuffle=True)

        # ---------------------------
        # Client encoder
        # ---------------------------
        input_dim = len(CLIENT_FEATURE_MAP[self.cid])
        self.model = ClientModel(input_dim=input_dim).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    # --------------------------------------------------------
    # Flower required methods
    # --------------------------------------------------------

    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for p in self.model.parameters()]

    def set_parameters(self, parameters):
        for p, new_p in zip(self.model.parameters(), parameters):
            p.data = torch.tensor(new_p, device=self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        # Gradients come from server in true VFL
        # Flower handles backward pass via returned gradients
        for batch in self.loader:
            x = batch[0].to(self.device)
            embeddings = self.model(x)

            # Store embeddings for server aggregation
            # No local loss here
            embeddings.backward(
                gradient=torch.ones_like(embeddings)
            )

            self.optimizer.step()
            self.optimizer.zero_grad()

        return self.get_parameters(config), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        # Clients do not evaluate in VFL
        return 0.0, 0, {}


# ------------------------------------------------------------
# CLIENT FACTORY
# ------------------------------------------------------------

def client_fn(cid: str):
    # full_df must be passed in from server/simulation
    from vertical_fl.data import FULL_DATAFRAME
    return FlowerVFLClient(cid, FULL_DATAFRAME)


app = fl.ClientApp(client_fn)
