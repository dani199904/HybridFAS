
# -------------------- Globals --------------------
best_val_acc = 0  # Initialized, will be updated by checkpoint

# -------------------- Checkpoint Paths (Updated for V2.5) --------------------
ckpt_path = "skt_live_ckpt_v2.5.pth"
best_model_path = "skt_live_best_model_v2.5.pth"

# -------------------- Loss & Optimizer --------------------
# Using pos_weight < 1.0 because Spoof (1) is the majority class in CelebA-Spoof
# This forces the model to pay more attention to the Live (0) samples
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.6]).to(device))
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-3)

# -------------------- Load Checkpoint if Exists --------------------
start_epoch = 1
if os.path.exists(ckpt_path):
    print(f"🔄 Found checkpoint '{ckpt_path}', resuming training...")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['opt'])
    start_epoch = ckpt['epoch'] + 1
    best_val_acc = ckpt.get('best_val_acc', 0)
    print(f"Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
else:
    print("No checkpoint found, starting training from scratch.")

# -------------------- Checkpoint Saving Function --------------------
def save_ckpt(epoch, model, optimizer, val_acc):
    global best_val_acc
    
    # Save regular/resume checkpoint
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'opt': optimizer.state_dict(),
        'best_val_acc': best_val_acc
    }, ckpt_path)
    
    # Save best model separately if validation accuracy improves
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model updated at epoch {epoch} with val_acc={val_acc:.4f}")

# -------------------- Training Loop --------------------
for epoch in range(start_epoch, 51):
    model.train()
    train_loss = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        imgs = imgs.to(device)
        labels = labels.to(device).unsqueeze(1).float() 

        preds = model(imgs)
        loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for Transformer stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_loss += loss.item() * imgs.size(0)

    train_loss /= len(train_loader.dataset)

    # ---------------- Validation ----------------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            outputs = model(imgs)

            probs = torch.sigmoid(outputs).view(-1).cpu().numpy()
            labels_np = labels.view(-1).cpu().numpy()

            all_preds.extend(probs)
            all_labels.extend(labels_np)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # compute_metrics function should handle (0=live, 1=spoof)
    acc, apcer, bpcer = compute_metrics(all_preds, all_labels)

    print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Acc: {acc:.4f} | APCER: {apcer:.4f} | BPCER: {bpcer:.4f}")

    # Save checkpoint (normal + best)
    save_ckpt(epoch, model, optimizer, acc)
