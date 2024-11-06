# from fastapi import APIRouter, Depends, HTTPException, status

# from ..models.user import User, UserSchema
# from .. import utils
# from ..utils.get_user import get_user
# from ..utils.auth.generate_verification_code import generate_verification_code
# from ..utils.auth.send_verification_email import send_verification_email
# from fastapi_jwt_auth import AuthJWT


# router = APIRouter()


# @router.get("/me", response_model=UserSchema)
# async def get_me(user_id: str = Depends(get_user)):
#     user = await User.get(str(user_id))
#     r_user = UserSchema(
#         username=user.username,
#         email=user.email,
#     )
#     return r_user


# @router.put("/me/update", response_model=UserSchema)
# async def update_me(
#     user_update: UserSchema,
#     user_id: str = Depends(get_user),
#     Authorize: AuthJWT = Depends(),
# ):
#     user = await User.get(str(user_id))

#     # Update user attributes
#     if user_update.username != user.username:
#         user_exists = await User.find_one(User.username == user_update.username)
#         if user_exists:
#             raise HTTPException(
#                 status_code=status.HTTP_409_CONFLICT,
#                 detail="Username already exists",
#             )
#         user.username = user_update.username
#     if user_update.email != user.email:
#         if not utils.is_valid_email(user_update.email):
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid email"
#             )
#         email_exists = await User.find_one(User.email == user_update.email)
#         if email_exists:
#             raise HTTPException(
#                 status_code=status.HTTP_409_CONFLICT,
#                 detail="Email already exists",
#             )
#         old_email = user.email
#         user.email = user_update.email
#         verification_code = generate_verification_code()
#         await send_verification_email(user.email, verification_code)
#         user.verification_code = verification_code
#         user.is_verified = False

#         # Log out the user if email is updated
#         Authorize.unset_jwt_cookies()

#     user_exists = await User.find_one(User.username == user_update.username)
#     email_exists = await User.find_one(User.email == user_update.email)

#     # Save the updated user back to the database
#     await user.save()

#     # Return the updated user as a response
#     return UserSchema(
#         username=user.username,
#         email=user.email,
#     )


# @router.put("/me/change-password")
# async def change_password(
#     old_password: str, new_password: str, user_id: str = Depends(get_user)
# ):
#     user = await User.get(str(user_id))

#     # Verify the old password
#     if not utils.verify_password(old_password, user.password):
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect old password",
#         )

#     # Update the password with the new one
#     user.password = utils.hash_password(new_password)

#     # Save the updated user back to the database
#     await user.save()

#     # Return the updated user as a response
#     return {"status": "password updated successfully"}


# @router.delete(
#     "/me/delete",
# )
# async def delete_me(user_id: str = Depends(get_user)):
#     user = await User.get(str(user_id))

#     # Delete the user from the database
#     await user.delete()

#     # Return the deleted user as a response
#     return {"status": "account successfully deleted"}
