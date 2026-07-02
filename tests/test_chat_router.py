import unittest
from unittest.mock import Mock, patch

from fastapi import HTTPException

from src.chat.router import reset_conversation
from src.user.models import UserProfile


class ChatRouterTest(unittest.TestCase):
    @patch.object(UserProfile, "get")
    def test_reset_conversation_keeps_profile_and_deletes_history(
        self,
        get_user_profile,
    ):
        rag = Mock()
        db = Mock()
        get_user_profile.return_value = UserProfile(
            user_id="reset-user"
        )

        response = reset_conversation(
            user_id="reset-user",
            rag=rag,
            db=db,
        )

        get_user_profile.assert_called_once_with("reset-user", db)
        rag.delete_conversation.assert_called_once_with("reset-user")
        self.assertEqual(
            response.model_dump(),
            {
                "user_id": "reset-user",
                "message": "대화 기록이 초기화되었습니다.",
            },
        )

    @patch.object(UserProfile, "get")
    def test_reset_conversation_rejects_unknown_user(
        self,
        get_user_profile,
    ):
        rag = Mock()
        get_user_profile.side_effect = HTTPException(
            status_code=404,
            detail="사용자를 찾을 수 없습니다.",
        )

        with self.assertRaises(HTTPException) as context:
            reset_conversation(
                user_id="missing-user",
                rag=rag,
                db=Mock(),
            )

        self.assertEqual(context.exception.status_code, 404)
        rag.delete_conversation.assert_not_called()


if __name__ == "__main__":
    unittest.main()
