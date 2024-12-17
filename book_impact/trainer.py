"""
This module contains all the low level components required for dispo model training process
"""

import datasets
import torch
import json
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class BertRegressorTraining:
    def __init__(self, data_dict=None, optimizer=None, loss_fn=None, model=None, training_args=None, config=None):
        """
        Args:
            data_dict (_type_): data_dict which has been pre-processed
            optimizer (_type_): optimizer to use for model
            training_args (dict, optional): configuration for training. Defaults to TRAIN_CONFIG.
        """
        self.optimizer = optimizer
        self.config = training_args
        self.data_dict = data_dict
        self.loss_fn = loss_fn
        self.model = model
        self.train_mape_history = []
        self.valid_mape_history = []
        self.train_loader = None
        self.valid_load = None
        self.config = config

    def prepare_data_loader(self):
        """
        Prepares dataloaders from data dictionary
        can be partially controlled from config file
        """
        self.train_load = DataLoader(
            self.data_dict["train"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=1,
            prefetch_factor=2,
        )
        self.valid_load = DataLoader(
            self.data_dict["test"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=1,
            prefetch_factor=2,
        )

    def train_model(self):
        """
        contains the actual training loop,
        everytime a best model is encountered using validation set,
        it is stored in model_save_path in config file.
        """
        EPOCHS = self.config["epochs"]
        best_mape = float("inf")

        # loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(self.config["loss_penalties"]))
        loss_fn = self.loss_fn

        for epoch in range(EPOCHS):
            batch_train_mape_history = []
            batch_val_mape_history = []

            self.model.to(self.config["device"])

            # training Loop : Epoch Level
            for batch in tqdm(self.train_load, desc=f"Epoch {epoch+1}/{EPOCHS} "):
                # train mode
                self.model.train()

                # organize batch
                batch["y"] = batch.pop("y")
                batch["input_ids"] = batch["input_ids"].squeeze(1)

                # to_device
                input_ids = batch["input_ids"].to(self.config["device"])
                attention_mask = batch["attention_mask"].to(self.config["device"])
                labels = batch["y"].to(self.config["device"])

                # model out and gradients
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)
                loss.backward()

                # update weights
                self.optimizer.step()
                self.optimizer.zero_grad()

                # record train mape
                batch_train_mape_history.append(loss_fn(outputs, labels))

            for batch in self.valid_load:  # Valid Loop
                self.model.eval()

                batch["y"] = batch.pop("y")
                batch["input_ids"] = batch["input_ids"].squeeze(1)

                input_ids = batch["input_ids"].to(self.config["device"])
                attention_mask = batch["attention_mask"].to(self.config["device"])
                labels = batch["y"].to(self.config["device"])

                # infer
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

                batch_val_mape_history.append(loss_fn(outputs, labels))

            # summarize epoch
            train_mape = sum(batch_train_mape_history) / len(batch_train_mape_history)
            val_mape = sum(batch_val_mape_history) / len(batch_val_mape_history)

            print(f"train_mape: {train_mape} | val_mape: {val_mape}")

            self.train_mape_history.append(train_mape)
            self.valid_mape_history.append(val_mape)

            # check if best model so far
            # if val_mape > best_mape:
            #     best_mape = val_mape
            #     self.best_model = self.model

    def get_best_model(self) -> torch.nn.Module:
        """Gets the best model encountered during the training

        Returns:
            torch.nn.Module:
        """
        return self.best_model

    def get_train_val_history(self):
        return self.train_mape_history, self.valid_mape_history
