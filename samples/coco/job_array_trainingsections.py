import argparse
import FK2018


def job_array(array_id):
    training_splits = [
        [40, 40, 40],
        [20, 50, 50],
        [50, 20, 50],
        [50, 50, 20],
        [20, 20, 80],
        [20, 80, 20],
        [80, 20, 20],
        [10, 20, 90],
        [20, 30, 70],
        [30, 40, 50],
    ]
    section1_epochs, section2_epochs, section3_epochs = training_splits[array_id]
    FK2018.train_nnet(
        section1_epochs=section1_epochs,
        section2_epochs=section1_epochs + section2_epochs,
        section3_epochs=section1_epochs + section2_epochs + section3_epochs,
        log_file=str(array_id),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass job array id")
    parser.add_argument("array_id", type=int, help="job array id")
    args = parser.parse_args()
    job_array(args.array_id)
