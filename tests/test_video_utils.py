import unittest

from RAG_Service.video_utils import extract_video_id, format_timestamp


class VideoUtilsTests(unittest.TestCase):
    def test_accepts_video_id_and_supported_youtube_urls(self):
        video_id = "RE2c_eH0Sq4"
        examples = [
            video_id,
            f"https://www.youtube.com/watch?v={video_id}&list=playlist",
            f"https://youtu.be/{video_id}?t=23",
            f"https://youtube.com/shorts/{video_id}",
            f"https://youtube.com/embed/{video_id}",
            f"youtube.com/live/{video_id}",
        ]

        for value in examples:
            with self.subTest(value=value):
                self.assertEqual(extract_video_id(value), video_id)

    def test_rejects_invalid_hosts_and_ids(self):
        for value in ("", "https://example.com/watch?v=RE2c_eH0Sq4", "too-short"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    extract_video_id(value)

    def test_formats_timestamp(self):
        self.assertEqual(format_timestamp(3723), "01:02:03")


if __name__ == "__main__":
    unittest.main()
