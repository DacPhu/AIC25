import {addAnswer, getAnswers} from "../services/answer.js";

export async function action({ request }) {
    const formData = await request.formData();
    const answer = Object.fromEntries(formData);
    return await addAnswer(answer);
}

export async function loader() {
    return await getAnswers();
}
