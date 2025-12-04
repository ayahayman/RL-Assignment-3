# main.py
import os
import sys
import argparse
import numpy as np
from train import train_ppo
from test import test_agent
from hyperparameter_tuning import hyperparameter_search
from utils import (set_seed, save_results, plot_training_curves, 
                   plot_test_results, plot_hyperparameter_comparison,
                   print_summary, create_results_directory)

def main():
    """Main execution function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PPO agents on Gymnasium environments')
    parser.add_argument('--envs', nargs='+', choices=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'Pendulum-v1', 'all'],
                        default=['all'], help='Environments to train (default: all)')
    parser.add_argument('--trials', type=int, default=5, help='Number of hyperparameter tuning trials (default: 5)')
    parser.add_argument('--trial-episodes', type=int, default=200, help='Episodes per trial (default: 200)')
    parser.add_argument('--episodes', type=int, default=2000, help='Final training episodes (default: 2000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--skip-tuning', action='store_true', help='Skip hyperparameter tuning and use defaults')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Environment list
    all_environments = [
        'CartPole-v1',
        'Acrobot-v1',
        'MountainCar-v0',
        'Pendulum-v1'
    ]
    
    # Select environments based on arguments
    if 'all' in args.envs:
        environments = all_environments
    else:
        environments = args.envs
    
    print(f"\nTraining on environments: {', '.join(environments)}")
    print(f"Hyperparameter trials: {args.trials}")
    print(f"Episodes per trial: {args.trial_episodes}")
    print(f"Final training episodes: {args.episodes}")
    print(f"Random seed: {args.seed}\n")
    
    # Create results directory
    results_dir = create_results_directory('ppo_results')
    print(f"Results will be saved to: {results_dir}\n")
    
    all_results = {}
    
    for env_name in environments:
        print(f"\n{'#'*80}")
        print(f"# Training PPO on {env_name}")
        print(f"{'#'*80}\n")
        
        # Step 1: Hyperparameter tuning
        if args.skip_tuning:
            print(f"Step 1: Using Default Hyperparameters (tuning skipped)")
            best_params = {
                'learning_rate': 0.0003,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'entropy_coef': 0.01,
                'batch_size': 64,
                'n_epochs': 10,
                'n_steps': 2048
            }
            tuning_results = []
            print(f"Parameters: {best_params}\n")
        else:
            print(f"Step 1: Hyperparameter Tuning")
            best_params, tuning_results = hyperparameter_search(
                env_name, 
                num_trials=args.trials,
                episodes_per_trial=args.trial_episodes
            )
            
            # Plot hyperparameter results
            hyperparam_plot_path = os.path.join(results_dir, f'{env_name}_hyperparameters.png')
            plot_hyperparameter_comparison(tuning_results, env_name, hyperparam_plot_path)
        
        # Step 2: Train final model with best parameters
        print(f"\nStep 2: Training Final Model")
        print(f"Using best parameters: {best_params}\n")
        agent, train_rewards, train_lengths = train_ppo(
            env_name, 
            best_params, 
            episodes=args.episodes,
            verbose=True
        )
        
        # Plot training curves
        training_plot_path = os.path.join(results_dir, f'{env_name}_training.png')
        plot_training_curves(train_rewards, train_lengths, env_name, training_plot_path)
        
        # Step 3: Save trained model
        model_path = os.path.join(results_dir, f'{env_name}_model.pth')
        agent.save(model_path)
        print(f"Model saved to {model_path}")
        
        # Step 4: Test the trained agent
        print(f"\nStep 3: Testing Agent (100 episodes)")
        test_rewards, test_lengths = test_agent(agent, env_name, num_tests=100, verbose=True)
        
        # Plot test results
        test_plot_path = os.path.join(results_dir, f'{env_name}_test.png')
        plot_test_results(test_rewards, test_lengths, env_name, test_plot_path)
        
        # Compile results
        results = {
            'environment': env_name,
            'best_hyperparameters': best_params,
            'tuning_results': tuning_results,
            'training_stats': {
                'final_train_reward': float(np.mean(train_rewards[-50:])),
                'final_train_length': float(np.mean(train_lengths[-50:]))
            },
            'test_stats': {
                'mean_reward': float(np.mean(test_rewards)),
                'std_reward': float(np.std(test_rewards)),
                'min_reward': float(np.min(test_rewards)),
                'max_reward': float(np.max(test_rewards)),
                'mean_length': float(np.mean(test_lengths)),
                'std_length': float(np.std(test_lengths)),
                'min_length': float(np.min(test_lengths)),
                'max_length': float(np.max(test_lengths))
            },
            'train_rewards': train_rewards,
            'train_lengths': train_lengths,
            'test_rewards': test_rewards,
            'test_lengths': test_lengths
        }
        
        all_results[env_name] = results
        
        # Save individual environment results
        env_results_path = os.path.join(results_dir, f'{env_name}_results.json')
        save_results(results, env_results_path)
        
        print(f"\n{'='*80}\n")
    
    # Save all results
    all_results_path = os.path.join(results_dir, 'all_results.json')
    save_results(all_results, all_results_path)
    
    # Print final summary
    print_summary(all_results)
    
    print(f"All results saved to: {results_dir}")

if __name__ == "__main__":
    main()