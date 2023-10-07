import dbConnection from "@/app/api/utils/mongodb.js";
import User from "@/app/api/utils/schema.js";
import { NextResponse } from "next/server";

dbConnection(process.env.NEXT_PUBLIC_MONGODB_URL);

export async function POST(request) {
  try {
    const body = await request.json();

    const { email, password } = body;

    const user = await User.findOne({ email: email.toLowerCase() });
    if (user) {
      return NextResponse.json(
        { error: "user already exist" },
        { status: 400 }
      );
    }

    const newuser = new User({
      email,
      password,
    });

    await newuser.save();

    return NextResponse.json({ message: "user created successfully" });
  } catch (error) {
    return NextResponse.json({ err: error.message }, { status: 500 });
  }
}
