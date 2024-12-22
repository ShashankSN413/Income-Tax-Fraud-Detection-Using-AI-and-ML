import pandas as pd
import numpy as np
import random

def generate_transaction_data(num_rows, fraud_rate=0.9):

    transaction_types = ['CASH_OUT', 'PAYMENT', 'TRANSFER', 'DEBIT', 'CASH_IN']

    data = {
        'step': [],
        'type': [],
        'amount': [],
        'nameOrig': [],
        'oldbalanceOrg': [],
        'newbalanceOrig': [],
        'nameDest': [],
        'oldbalanceDest': [],
        'newbalanceDest': [],
        'isFraud': [],
        'isFlaggedFraud': [],
        'transaction_time': [],
        'transaction_location': [],
        'noise_feature': []
    }

    for _ in range(num_rows):

        step = random.randint(1, 30)
        t_type = random.choice(transaction_types)
        amount = round(random.uniform(0.01, 100000.00), 2)
        name_orig = f"C{random.randint(1000000000, 9999999999)}"
        old_balance_orig = round(random.uniform(0.00, 100000.00), 2)
        new_balance_orig = max(0, old_balance_orig - amount)
        name_dest = f"C{random.randint(1000000000, 9999999999)}" if t_type in ['TRANSFER', 'CASH_OUT'] else f"M{random.randint(1000000000, 9999999999)}"
        old_balance_dest = round(random.uniform(0.00, 100000.00), 2)
        new_balance_dest = old_balance_dest + amount if t_type == 'TRANSFER' else old_balance_dest
        is_fraud = 1 if t_type in ['TRANSFER', 'CASH_OUT'] and random.random() < fraud_rate else 0
        is_flagged_fraud = 1 if is_fraud and amount > 200000 else 0

        transaction_time = random.choice(['Morning', 'Afternoon', 'Evening', 'Night'])
        transaction_location = random.choice(['US', 'EU', 'Asia', 'Africa'])

        noise_feature = random.uniform(0, 1)

        data['step'].append(step)
        data['type'].append(t_type)
        data['amount'].append(amount)
        data['nameOrig'].append(name_orig)
        data['oldbalanceOrg'].append(old_balance_orig)
        data['newbalanceOrig'].append(new_balance_orig)
        data['nameDest'].append(name_dest)
        data['oldbalanceDest'].append(old_balance_dest)
        data['newbalanceDest'].append(new_balance_dest)
        data['isFraud'].append(is_fraud)
        data['isFlaggedFraud'].append(is_flagged_fraud)
        data['transaction_time'].append(transaction_time)
        data['transaction_location'].append(transaction_location)
        data['noise_feature'].append(noise_feature)

    df = pd.DataFrame(data)
    return df

num_rows = int(input("Enter the number of rows to generate: "))
fraud_rate = float(input("Enter the fraud rate (e.g., 0.9 for 90%): "))

fraud_dataset = generate_transaction_data(num_rows, fraud_rate)

file_name = "synthetic_fraud_dataset_with_noise.csv"
fraud_dataset.to_csv(file_name, index=False)

print(f"Dataset generated and saved as {file_name}")