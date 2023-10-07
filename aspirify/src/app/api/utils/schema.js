import mongoose from "mongoose";
import bcrypt from "bcryptjs";

const credential = new mongoose.Schema({
  email: {
    type: String,
    required: [true, "please provide an email"],
    unique: true,
  },
  password: {
    type: String,
    required: [true, "please provide a password"],
  },
});
credential.pre("save", async function () {
  this.password = await bcrypt.hash(this.password, 10);
});
const User = mongoose.models.users || mongoose.model("users", credential);

export default User;
