import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        
        # FIX: Start gates at 0.0 so the optimizer has a head start to reach the prune threshold
        nn.init.constant_(self.gate_scores, 0.0) 

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def get_sparsity(self, threshold=1e-2):
        gates = torch.sigmoid(self.gate_scores)
        pruned_count = (gates < threshold).sum().item()
        total_count = gates.numel()
        return pruned_count, total_count

class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def exec_training_epoch(model, device, loader, opt, penalty_lambda):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        opt.zero_grad()
        
        preds = model(data)
        cls_loss = F.cross_entropy(preds, target)
        
        l1_penalty = sum(
            torch.sum(torch.sigmoid(m.gate_scores)) 
            for m in model.modules() if isinstance(m, PrunableLinear)
        )
        
        loss = cls_loss + penalty_lambda * l1_penalty
        loss.backward()
        opt.step()

def calc_metrics(model, device, loader):
    model.eval()
    correct_preds = 0
    pruned_total = 0
    param_total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            preds = model(data)
            pred_classes = preds.argmax(dim=1, keepdim=True)
            correct_preds += pred_classes.eq(target.view_as(pred_classes)).sum().item()
    
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            p_count, t_count = m.get_sparsity()
            pruned_total += p_count
            param_total += t_count
            
    acc = 100. * correct_preds / len(loader.dataset)
    sparsity = 100. * pruned_total / param_total
    return acc, sparsity

if __name__ == "__main__":
    compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    img_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=img_transforms)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=img_transforms)
    
    loader_train = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    loader_test = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False)

    # FIX: Heavier penalties to force the network to prune
    test_lambdas = [0.001, 0.01, 0.05] 
    print(f"{'Penalty (Lambda)':<18} | {'Accuracy (%)':<15} | {'Sparsity (%)':<15}")
    print("-" * 55)
    
    for p_lambda in test_lambdas:
        net = PrunableNet().to(compute_device)
        
        # FIX: Higher learning rate so gates reach zero within 10 epochs
        optimizer = optim.Adam(net.parameters(), lr=0.005) 
        
        for _ in range(10): 
            exec_training_epoch(net, compute_device, loader_train, optimizer, p_lambda)
        
        final_acc, final_spar = calc_metrics(net, compute_device, loader_test)
        print(f"{p_lambda:<18} | {final_acc:<15.2f} | {final_spar:<15.2f}")

    # Extract gates from the final trained model
    all_gates = []
    for m in net.modules():
        if isinstance(m, PrunableLinear):
            gates = torch.sigmoid(m.gate_scores).detach().cpu().numpy().flatten()
            all_gates.extend(gates)
    
    # Generate and save the plot explicitly to your Mac's Desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'gate_distribution.png')
    
    plt.figure(figsize=(8, 5))
    plt.hist(all_gates, bins=50, color='royalblue', edgecolor='black')
    plt.title('Distribution of Final Gate Values')
    plt.xlabel('Gate Value (0 = Pruned, 1 = Kept)')
    plt.ylabel('Number of Weights')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(desktop_path)
    print(f"Plot successfully saved directly to your Desktop at: {desktop_path}")
