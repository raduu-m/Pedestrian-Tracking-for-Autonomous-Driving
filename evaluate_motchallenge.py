# vim: expandtab:ts=4:sw=4
import argparse
import os
import deep_sort_app


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="MOTChallenge evaluation")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to detections.", default="detections",
        required=True)
    parser.add_argument(
        "--output_dir", help="Folder in which the results will be stored. Will "
        "be created if it does not exist.", default="results")
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value. Set to "
        "0.3 to reproduce results in the paper.",
        default=0.3, type=float)
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maximum suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=100)
    parser.add_argument(
        "--use_bloom_filter", help="Enable Bloom filter for tracking optimization",
        action="store_true", default=True)
    parser.add_argument(
        "--expected_tracks", help="Expected number of tracks for Bloom filter sizing",
        type=int, default=1000)
    parser.add_argument(
        "--bloom_false_positive_rate", help="Bloom filter false positive rate",
        type=float, default=0.01)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sequences = os.listdir(args.mot_dir)
    for sequence in sequences:
        print("Running sequence %s" % sequence)
        sequence_dir = os.path.join(args.mot_dir, sequence)
        detection_file = os.path.join(args.detection_dir, "%s.npy" % sequence)
        output_file = os.path.join(args.output_dir, "%s.txt" % sequence)
        deep_sort_app.run(
            sequence_dir, detection_file, output_file, args.min_confidence,
            args.nms_max_overlap, args.min_detection_height,
            args.max_cosine_distance, args.nn_budget, display=False,
            use_bloom_filter=args.use_bloom_filter,
            expected_tracks=args.expected_tracks,
            bloom_false_positive_rate=args.bloom_false_positive_rate)
