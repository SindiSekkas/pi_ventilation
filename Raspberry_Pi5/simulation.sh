#!/bin/bash
# simulation.sh - Script to run ventilation simulation with different modes

# Usage information
function show_help {
    echo "Usage: ./simulation.sh [command] [options]"
    echo "Commands:"
    echo "  train       Train the Markov model (long-term simulation)"
    echo "  evaluate    Run evaluation with all strategies (using pre-trained model)"
    echo "  compare     Compare previously generated results"
    echo ""
    echo "Options:"
    echo "  --duration DAYS    Duration in days (default: 90 for train, 7 for evaluate)"
    echo "  --output-dir DIR   Output directory (default: simulation_results)"
    echo "  --time-step MINS   Time step in minutes (default: 5)"
    echo "  --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./simulation.sh train --duration 90"
    echo "  ./simulation.sh evaluate --duration 14"
    echo "  ./simulation.sh evaluate --strategies threshold markov"
}

# Default values
DURATION=""
OUTPUT_DIR=""
TIME_STEP=""
STRATEGIES="all"

# Parse command
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

COMMAND=$1
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="--duration $2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="--output-dir $2"
            shift 2
            ;;
        --time-step)
            TIME_STEP="--time-step $2"
            shift 2
            ;;
        --strategies)
            STRATEGIES="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Execute the requested command
case $COMMAND in
    train)
        echo "Training Markov model..."
        if [ -z "$DURATION" ]; then
            DURATION="--duration 90"
        fi
        python train_markov_model.py $DURATION $OUTPUT_DIR $TIME_STEP
        ;;
    evaluate)
        echo "Running evaluation with all strategies..."
        if [ -z "$DURATION" ]; then
            DURATION="--duration 7"
        fi
        if [ "$STRATEGIES" == "all" ]; then
            python run_simulation.py $DURATION $OUTPUT_DIR $TIME_STEP --use-pretrained
        else
            python run_simulation.py $DURATION $OUTPUT_DIR $TIME_STEP --use-pretrained --strategies $STRATEGIES
        fi
        ;;
    compare)
        echo "Comparing results..."
        python run_simulation.py --compare-only
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac

echo "Done!"